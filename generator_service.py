import os
import json
import uuid
import re
import threading
import traceback
import shutil
import gc
import warnings
import queue
import logging
import ctypes
import time
import psutil
from collections import Counter
from types import SimpleNamespace
from flask import Flask, request, jsonify
import torch

# --- НАСТРОЙКА ЛОГИРОВАНИЯ ---
logging.basicConfig(
    level=logging.INFO,
    format='[GEN] %(message)s'
)
logger = logging.getLogger(__name__)

YANDEX_REQUEST_MIN_INTERVAL_SECONDS = 0.85
YANDEX_REQUEST_FAILURE_COOLDOWN_SECONDS = 1.5

ALBUM_TRACK_RE = re.compile(r"album/(\d+)/track/(\d+)", re.IGNORECASE)
TRACK_ONLY_RE = re.compile(r"track/(\d+)", re.IGNORECASE)


def extract_yandex_track_identity(url):
    if not url:
        return None

    normalized = str(url).strip()
    if not normalized:
        return None

    album_track_match = ALBUM_TRACK_RE.search(normalized)
    if album_track_match:
        return ("album_track", album_track_match.group(1), album_track_match.group(2))

    track_only_match = TRACK_ONLY_RE.search(normalized)
    if track_only_match:
        return ("track", track_only_match.group(1))

    return None


def _identity_label(identity, raw_url):
    if not identity:
        return raw_url.strip()
    if identity[0] == "album_track":
        return f"album/{identity[1]}/track/{identity[2]}"
    if identity[0] == "track":
        return f"track/{identity[1]}"
    return raw_url.strip()


def dedupe_yandex_track_urls(urls):
    unique_urls = []
    duplicates = []
    seen = set()

    for raw_url in urls or []:
        url = str(raw_url).strip()
        if not url:
            continue

        identity = extract_yandex_track_identity(url)
        key = identity if identity else ("raw", url)
        if key in seen:
            duplicates.append(_identity_label(identity, url))
            continue

        seen.add(key)
        unique_urls.append(url)

    return unique_urls, duplicates


class YandexRequestThrottle:
    def __init__(self, min_interval_seconds=0.45, time_fn=None, sleep_fn=None):
        self.min_interval_seconds = max(0.0, float(min_interval_seconds))
        self._time_fn = time_fn or time.time
        self._sleep_fn = sleep_fn or time.sleep
        self._next_request_at = None

    def wait(self):
        now = self._time_fn()
        if self._next_request_at is None:
            self._next_request_at = now + self.min_interval_seconds
            return 0.0

        if now < self._next_request_at:
            delay = self._next_request_at - now
            self._sleep_fn(delay)
            now = self._next_request_at
            self._next_request_at = now + self.min_interval_seconds
            return delay

        self._next_request_at = now + self.min_interval_seconds
        return 0.0

    def penalize(self, extra_delay_seconds):
        extra_delay_seconds = max(0.0, float(extra_delay_seconds))
        if extra_delay_seconds == 0.0:
            return 0.0

        now = self._time_fn()
        base = self._next_request_at if self._next_request_at is not None else now
        self._next_request_at = max(base, now) + extra_delay_seconds
        return self._next_request_at - now

# Отключаем шумные логи библиотек
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
logging.getLogger("whisperx").setLevel(logging.WARNING)

# Отключаем UserWarning от torchaudio, pyannote и speechbrain
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
warnings.filterwarnings("ignore", message=".*torchaudio._backend.*")

# --- ПАТЧ PYTORCH (для совместимости с audio-separator) ---
try:
    _original_load = torch.load
    def _safe_load_wrapper(*args, **kwargs):
        if 'weights_only' in kwargs:
            del kwargs['weights_only']
        return _original_load(*args, weights_only=False, **kwargs)
    torch.load = _safe_load_wrapper
    try:
        from omegaconf import listconfig
        torch.serialization.add_safe_globals([listconfig.ListConfig])
    except ImportError:
        pass  # omegaconf не установлен — не критично
    logger.info("✅ PyTorch patched!")
except Exception as e:
    logger.error(f"❌ Patch failed: {e}")

import whisperx
from pydub import AudioSegment
from yandex_music import Client as YandexClient
from yandex_music.exceptions import NotFoundError

app = Flask(__name__)

# --- СЛОВАРИ И ФИЛЬТРЫ ---

# --- ФИЛЬТР МУСОРА WHISPER ---
# Мусор всегда приходит как ОТДЕЛЬНЫЙ сегмент целиком.
# Нормализуем текст сегмента и сравниваем с эталонами.
JUNK_SEGMENTS = {
    # Технические надписи / титры
    "редактор субтитров", "субтитры", "корректор",
    "подогнал", "dimatorzok",
    # Галлюцинации Whisper (русские)
    "продолжение следует", "конец фильма", "спасибо за просмотр",
    "не забудьте подписаться", "подписывайтесь",
    "ставьте лайки", "ставьте лайк",
    "следующее видео", "предыдущее видео",
    "нажмите колокольчик", "в следующий раз",
    "спасибо что смотрите", "спасибо",
    "приятного просмотра", "до новых встреч",
    "оставайтесь с нами", "смотрите также",
    "добро пожаловать", "до свидания", "всем привет",
    # Английские
    "like and subscribe", "thanks for watching",
    "see you next time", "thank you for watching",
    "thank you", "subscribe", "subtitles", "translated",
    # Артефакты из промпта Whisper
    "текст песни", "песня на русском", "русском языке",
    "рифма", "куплеты", "припев", "поэзия",
    "рифма куплеты припев",
    "текст песни на русском языке",
    "текст песни на русском языке рифма куплеты припев",
    # Мычание / бессмыслица
    "ля ля ля", "на на на",
}


def _normalize_for_junk_check(text):
    """Оставляет только буквы и пробелы, схлопывает пробелы."""
    text = text.lower().strip()
    text = re.sub(r'[^а-яёa-z ]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def is_junk_segment(segment_text):
    """
    Проверяет, является ли ВЕСЬ сегмент мусором Whisper.
    Мусор всегда приходит как отдельный сегмент.
    """
    if not segment_text or not segment_text.strip():
        return True

    norm = _normalize_for_junk_check(segment_text)
    if not norm:
        return True

    # 1. Точное совпадение или сегмент начинается с известного мусора
    #    (Whisper может дописать имена: "редактор субтитров асемкин корректор аегорова")
    if norm in JUNK_SEGMENTS:
        return True
    for junk in JUNK_SEGMENTS:
        if len(junk) >= 10 and norm.startswith(junk):
            return True

    # 2. Повторение одного слова (ааа ааа ааа ааа)
    words = norm.split()
    if len(words) > 3 and len(set(words)) == 1:
        return True

    # 3. Слишком длинный сегмент (галлюцинация-простыня)
    if len(norm) > 500:
        return True

    # 4. >90% одинаковых слов
    if len(words) > 5:
        word_counts = Counter(words)
        most_common = word_counts.most_common(1)
        if most_common and most_common[0][1] / len(words) > 0.9:
            return True

    return False

def clean_lrc_lyrics(lrc_text):
    """
    Очищает LRC текст от таймкодов.
    Формат LRC: [00:15.50]Текст строки
    Возвращает чистый текст без таймкодов.
    """
    if not lrc_text:
        return ""
    # Убираем таймкоды вида [00:15.50] или [00:15]
    clean_text = re.sub(r'\[\d{2}:\d{2}(?:\.\d{2,3})?\]', '', lrc_text)
    # Убираем пустые строки и лишние пробелы
    lines = [line.strip() for line in clean_text.split('\n') if line.strip()]
    return '\n'.join(lines)


def align_words_to_lyrics(all_words, official_lyrics):
    """
    Совмещает Whisper-слова (точные таймкоды, неточный текст) с official lyrics
    (точный текст, нет таймкодов).

    Алгоритм: проходим по строкам lyrics и по Whisper-словам параллельно.
    Для каждого lyrics-слова ищем ближайшее Whisper-слово (по нечёткому сравнению
    и позиции). Заменяем текст Whisper-слова на lyrics-слово, сохраняя таймкоды.

    Результат: all_words с исправленным текстом + is_eol маркерами из lyrics.
    """
    if not all_words or not official_lyrics:
        return all_words

    # Разбиваем lyrics на строки → слова
    lyrics_lines = [l.strip() for l in official_lyrics.split('\n') if l.strip()]
    lyrics_flat = []  # [(clean_word, original_word, line_idx, is_last_in_line)]
    for li, line in enumerate(lyrics_lines):
        line_words = re.findall(r'[а-яёА-ЯЁa-zA-Z\-]+', line)
        for wi, w in enumerate(line_words):
            is_last = (wi == len(line_words) - 1)
            lyrics_flat.append((clean_word(w), w, li, is_last))

    if not lyrics_flat:
        return all_words

    log(f"🔗 Align: {len(all_words)} Whisper-слов ↔ {len(lyrics_flat)} lyrics-слов ({len(lyrics_lines)} строк)")

    # --- Жадное сопоставление с нечётким сравнением ---
    # Идём по lyrics_flat, для каждого слова ищем ближайший match в Whisper
    # с ограничением: Whisper-индекс может только расти (монотонность)
    whisper_idx = 0
    matched_count = 0
    pending_skipped = []  # Lyrics-слова которые Whisper пропустил (ещё не привязаны)
    result_words = []  # Копия all_words с заменённым текстом

    # Копируем all_words
    for w in all_words:
        result_words.append(dict(w))

    for lyrics_idx, (lyr_clean, lyr_orig, line_idx, is_last_in_line) in enumerate(lyrics_flat):
        if whisper_idx >= len(all_words):
            break

        # Ищем лучший match в окне [whisper_idx, whisper_idx + search_window]
        # Окно шире для первых слов строки (может быть пропуск слов Whisper'ом)
        search_window = 5
        best_match_idx = -1
        best_score = 0.0

        for wi in range(whisper_idx, min(whisper_idx + search_window, len(all_words))):
            w_clean = clean_word(all_words[wi]['word'])

            # Точное совпадение — идеально
            if w_clean == lyr_clean:
                best_match_idx = wi
                best_score = 1.0
                break

            # Нечёткое сравнение
            ratio = _levenshtein_ratio(w_clean, lyr_clean)
            # Порог: минимум 60% совпадения и длина слова >= 3
            if ratio > best_score and ratio >= 0.55 and len(lyr_clean) >= 3:
                best_match_idx = wi
                best_score = ratio

        if best_match_idx >= 0:
            # Заменяем текст Whisper на lyrics
            old_text = result_words[best_match_idx]['word']
            result_words[best_match_idx]['word'] = lyr_orig
            result_words[best_match_idx]['is_eol'] = is_last_in_line
            result_words[best_match_idx]['lyrics_line_idx'] = line_idx
            result_words[best_match_idx]['lyrics_line'] = lyrics_lines[line_idx]
            # Сохраняем пропущенные lyrics-слова, которые Whisper не увидел
            # (накопились между предыдущим match и этим)
            if pending_skipped:
                result_words[best_match_idx]['skipped_before'] = list(pending_skipped)
                pending_skipped.clear()
            if best_score < 1.0:
                log(f"  🔄 '{old_text}' → '{lyr_orig}' (match={best_score:.0%})")
            matched_count += 1
            whisper_idx = best_match_idx + 1
        else:
            # Lyrics-слово не найдено в Whisper — запоминаем как пропущенное
            pending_skipped.append(lyr_orig)

    match_pct = matched_count / len(lyrics_flat) * 100 if lyrics_flat else 0
    log(f"🔗 Align результат: {matched_count}/{len(lyrics_flat)} слов совмещено ({match_pct:.0f}%)")

    # Если совмещение слишком плохое (<40%), не доверяем — вернём оригинал
    if match_pct < 40:
        log(f"⚠️ Align: слишком мало совпадений ({match_pct:.0f}%), используем оригинальный Whisper текст")
        return all_words

    return result_words


STOP_WORDS = {
    "и", "в", "во", "не", "на", "я", "с", "со", "он", "она", "оно", "они",
    "а", "но", "к", "у", "по", "из", "за", "от", "о", "об", "для", "до", 
    "же", "ну", "вы", "мы", "ты", "бы", "ли", "или", "тут", "там", "где", 
    "как", "что", "кто", "это", "то", "так", "вот", "все", "всё", "уже", 
    "еще", "ещё", "только", "просто", "потому", "когда", "если", "мой", 
    "твой", "свой", "наш", "ваш", "меня", "тебя", "себя", "его", "ее", "их",
    "было", "будет", "есть", "нет", "да", "был", "была", "были", "пусть",
    "даже", "раз", "два", "три", "теперь", "сейчас", "потом", "тогда",
    "здесь", "через", "очень", "надо", "может", "наверное", "конечно",
    "тебе", "себе", "мной", "тобой", "этом", "этой", "того", "чего",
    "куда", "туда", "сюда", "быть", "стал", "стала", "стали"    
}

# --- КОНФИГУРАЦИЯ ---
BASE_DIR = os.getcwd()
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
WHISPER_SIZE = "large-v3"
COMPUTE_TYPE = "int8"
DEFAULT_CPU_MDX_BATCH_SIZE = 1
DEFAULT_CPU_WHISPER_THREADS = 8
DEFAULT_CPU_WHISPER_BATCH_SIZE = 8
DEFAULT_GPU_MDX_BATCH_SIZE = 8
DEFAULT_GPU_WHISPER_THREADS = 8
DEFAULT_GPU_WHISPER_BATCH_SIZE = 16

# DSP настройки (ревью: ослаблены для лучшего качества)
DSP_HIGHPASS_FREQ = 100          # Было 200Hz — слишком агрессивно для мужского вокала
DSP_COMPRESSOR_RATIO = 2.5       # Было 4.0 — слишком жёстко, поднимает шумы
DSP_COMPRESSOR_THRESHOLD = -20.0

# Тайминги викторины
MIN_QUESTION_DURATION_MS = 20000  # Минимальная длительность вопроса
TARGET_QUESTION_DURATION_MS = 28000
MIN_AUDIO_POSITION = 12.0         # Минимальная позиция слова в секундах (spoiler protection)
QUESTION_WORD_GUARD_MS = 100

# Scoring для алгоритмического выбора (score_candidates)
SCORE_EOL_BONUS = 80              # Бонус за конец строки (рифма) — ГЛАВНЫЙ ПРИОРИТЕТ
SCORE_PUNCTUATION_BONUS = 20      # Бонус за пунктуацию
SCORE_LONG_WORD_BONUS = 10        # Бонус за длинное слово (>=6 букв)
SCORE_MEDIUM_WORD_BONUS = 5       # Бонус за среднее слово (>=5 букв)
SCORE_UNIQUE_WORD_BONUS = 25      # Бонус за уникальное слово
SCORE_RARE_WORD_BONUS = 15        # Бонус за редкое слово (2 вхождения)
SCORE_FREQUENT_PENALTY = -50      # Штраф за частое слово (>=4 вхождений)
SCORE_VERB_ENDING_PENALTY = -10   # Штраф за глагольные окончания

ALLOWED_SINGLE_CHAR_WORDS = {"а", "и", "в", "к", "с", "у", "о", "я"}
RUSSIAN_VOWELS = set("аеёиоуыэюя")
QUESTION_WORD_BLOCKING_SEPARATORS = "-."

if not os.path.exists(MEDIA_ROOT):
    os.makedirs(MEDIA_ROOT)


def _env_int(name, default, min_value=1):
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default

    raw_value = str(raw_value).strip()
    if not raw_value:
        return default

    try:
        parsed = int(raw_value)
    except ValueError:
        return default

    if parsed < min_value:
        return default
    return parsed


def get_runtime_tuning(device):
    is_gpu = str(device).lower() == "cuda"
    if is_gpu:
        defaults = {
            "mdx_batch_size": DEFAULT_GPU_MDX_BATCH_SIZE,
            "whisper_threads": DEFAULT_GPU_WHISPER_THREADS,
            "whisper_batch_size": DEFAULT_GPU_WHISPER_BATCH_SIZE,
        }
    else:
        defaults = {
            "mdx_batch_size": DEFAULT_CPU_MDX_BATCH_SIZE,
            "whisper_threads": DEFAULT_CPU_WHISPER_THREADS,
            "whisper_batch_size": DEFAULT_CPU_WHISPER_BATCH_SIZE,
        }

    return {
        "mdx_batch_size": _env_int("MDX_BATCH_SIZE", defaults["mdx_batch_size"]),
        "whisper_threads": _env_int("WHISPER_THREADS", defaults["whisper_threads"]),
        "whisper_batch_size": _env_int("WHISPER_BATCH_SIZE", defaults["whisper_batch_size"]),
    }

# --- СОСТОЯНИЕ ЗАДАЧ (thread-safe) ---
job_status_lock = threading.Lock()
job_status = {
    "is_busy": False, "progress": 0, "logs": [], "status": "idle"
}

# Очередь задач (вместо создания потока на каждый запрос)
task_queue = queue.Queue()


def task_worker():
    """
    Worker-поток, обрабатывающий задачи из очереди.
    Один поток на всё время жизни приложения — нет накладных расходов на создание.
    """
    while True:
        task_data = task_queue.get()
        try:
            generation_task(
                task_data['game_id'],
                task_data.get('token'),
                task_data.get('urls', [])
            )
        except Exception as e:
            logger.error(f"❌ Worker error: {e}\n{traceback.format_exc()}")
            with job_status_lock:
                job_status["status"] = "error"
                job_status["is_busy"] = False
        finally:
            task_queue.task_done()


_gen_start_time = 0.0  # Время старта текущей генерации

def log(msg):
    """Thread-safe логирование с временной меткой."""
    elapsed = time.time() - _gen_start_time if _gen_start_time > 0 else 0
    timestamp = f"[{elapsed:6.1f}s]" if elapsed > 0 else "[     ]"
    with job_status_lock:
        job_status["logs"].append(f"{timestamp} {msg}")
    logger.info(f"{timestamp} {msg}")

# --- АУДИО УТИЛИТЫ ---

def preprocess_for_whisper(audio_path):
    """
    Подготовка аудио: 16kHz + Компрессия + HighPass фильтр.
    Делает вокал 'плотным' и удаляет остатки басов.
    """
    try:
        from pydub.effects import normalize, compress_dynamic_range

        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_frame_rate(16000).set_channels(1)

        # 1. High-pass фильтр (удаляем гул, сохраняя мужской бас ~80-100Hz)
        # Ревью: 200Hz было слишком агрессивно — делало голос "телефонным"
        audio = audio.high_pass_filter(DSP_HIGHPASS_FREQ)

        # 2. Агрессивная нормализация перед компрессией
        audio = normalize(audio)
        
        # 3. Динамическая компрессия (выравнивает громкость)
        # Ревью: ratio 4.0 было слишком жёстко — поднимало шумы до уровня речи
        # Теперь ratio=2.5 — достаточно для вокала без артефактов
        audio = compress_dynamic_range(
            audio,
            threshold=DSP_COMPRESSOR_THRESHOLD,
            ratio=DSP_COMPRESSOR_RATIO,
            attack=5.0,
            release=50.0
        )
        
        # 4. Финальная нормализация
        audio = normalize(audio)

        output_path = audio_path.replace('.wav', '_proc.wav')
        # Если расширение было не wav, заменим корректно
        if output_path == audio_path:
             output_path = os.path.splitext(audio_path)[0] + '_proc.wav'

        audio.export(output_path, format='wav')
        log(f"🎵 DSP обработка завершена (HighPass + Compressor)")
        return output_path

    except Exception as e:
        log(f"⚠️ Ошибка DSP: {e}")
        return audio_path

# --- КЭШ МОДЕЛИ РАЗДЕЛЕНИЯ ---
_cached_separator = None

def release_separator():
    """Освобождает кэшированный Separator."""
    global _cached_separator
    if _cached_separator is not None:
        del _cached_separator
        _cached_separator = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log("🗑️ Separator освобождён из памяти")


def isolate_vocals(audio_path, use_cache=True):
    """
    Изоляция вокала с помощью Roformer (mel_band_roformer).
    """
    global _cached_separator
    tuning = get_runtime_tuning("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = os.path.dirname(audio_path)
    filename = os.path.basename(audio_path)
    file_id = os.path.splitext(filename)[0]

    expected_vocals = os.path.join(output_dir, f"{file_id}_vocals.wav")
    if use_cache and os.path.exists(expected_vocals):
        log(f"✅ Кэш вокала: {os.path.basename(expected_vocals)}")
        return expected_vocals

    log(f"🎸 Запуск Roformer для {filename}...")
    t0 = time.time()

    try:
        from audio_separator.separator import Separator
        logging.getLogger("audio_separator").setLevel(logging.ERROR)
        model_cache_dir = os.environ.get("MDX_MODEL_CACHE", "/app/mdx_cache")

        if _cached_separator is None:
            _cached_separator = Separator(
                log_level=logging.ERROR,
                output_dir=output_dir,
                output_format="wav",
                model_file_dir=model_cache_dir,
                output_single_stem="Vocals",
                mdxc_params={"batch_size": tuning["mdx_batch_size"]}
            )
            _cached_separator.load_model(
                model_filename="model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt"
            )
        else:
            _cached_separator.output_dir = output_dir

        output_files = _cached_separator.separate(audio_path)
        elapsed = time.time() - t0

        generated_file = None
        if output_files and len(output_files) > 0:
            generated_file = os.path.join(output_dir, output_files[0])

        if generated_file and os.path.exists(generated_file):
            if os.path.exists(expected_vocals):
                os.remove(expected_vocals)
            os.rename(generated_file, expected_vocals)
            log(f"✅ Вокал извлечён за {elapsed:.1f}с: {os.path.basename(expected_vocals)}")
            return expected_vocals
        else:
            log("⚠️ Separator не создал файл, используем оригинал")
            return audio_path

    except Exception as e:
        log(f"⚠️ Ошибка Roformer: {e}")
        return audio_path


def _log(logger, message):
    if logger:
        logger(message)


def clean_word(word):
    word = (word or "").lower().replace("ё", "е")
    return re.sub(r"[^\w]", "", word)


def _word_has_vowel(word):
    return any(char in RUSSIAN_VOWELS for char in word)


def _should_preserve_skipped_token(token):
    clean = clean_word(token)
    if not clean or not re.fullmatch(r"[а-яё]+", clean):
        return False
    if clean in ALLOWED_SINGLE_CHAR_WORDS or clean in STOP_WORDS:
        return True
    return 2 <= len(clean) <= 5 and _word_has_vowel(clean)


def _clean_lyrics_line_text(text):
    text = (text or "").replace("\xa0", " ")
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\[[^\]]*\]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _line_words(text):
    return re.findall(r"[а-яёА-ЯЁa-zA-Z0-9\-]+", _clean_lyrics_line_text(text))


def _visible_text_tokens(text):
    return re.findall(r"\S+", _clean_lyrics_line_text(text))


def extract_separator_blocked_fragments(text, separators=QUESTION_WORD_BLOCKING_SEPARATORS):
    if not text:
        return set()

    separator_class = re.escape(separators)
    blocked = set()
    for token in _visible_text_tokens(text):
        if not re.search(rf"\w[{separator_class}]\w", token, flags=re.UNICODE):
            continue

        parts = [
            clean_word(part)
            for part in re.split(rf"[{separator_class}]+", token)
            if clean_word(part)
        ]
        if len(parts) >= 2:
            blocked.update(parts)

    return blocked


def is_separator_fragment_candidate(word, separators=QUESTION_WORD_BLOCKING_SEPARATORS):
    clean = clean_word(word.get("word", ""))
    if not clean:
        return False

    source_texts = [
        word.get("lyrics_line", ""),
        word.get("segment_text", ""),
    ]
    for source_text in source_texts:
        if clean in extract_separator_blocked_fragments(source_text, separators=separators):
            return True

    return False


def _levenshtein_ratio(left, right):
    if not left or not right:
        return 0.0
    if left == right:
        return 1.0

    len_left = len(left)
    len_right = len(right)
    previous = list(range(len_right + 1))
    for left_idx in range(1, len_left + 1):
        current = [left_idx] + [0] * len_right
        for right_idx in range(1, len_right + 1):
            cost = 0 if left[left_idx - 1] == right[right_idx - 1] else 1
            current[right_idx] = min(
                current[right_idx - 1] + 1,
                previous[right_idx] + 1,
                previous[right_idx - 1] + cost,
            )
        previous = current
    return 1.0 - (previous[len_right] / max(len_left, len_right))


def _token_overlap(words_a, words_b):
    if not words_a or not words_b:
        return 0.0

    set_a = set(words_a)
    set_b = set(words_b)
    return len(set_a & set_b) / max(len(set_a), len(set_b))


def _line_similarity(line_a, line_b):
    words_a = [clean_word(word) for word in _line_words(line_a) if clean_word(word)]
    words_b = [clean_word(word) for word in _line_words(line_b) if clean_word(word)]
    if not words_a or not words_b:
        return 0.0

    text_ratio = _levenshtein_ratio(" ".join(words_a), " ".join(words_b))
    overlap_ratio = _token_overlap(words_a, words_b)
    return text_ratio * 0.6 + overlap_ratio * 0.4


def _collect_line_records(all_words):
    records = []
    current = None
    current_line_idx = None

    for idx, word in enumerate(all_words):
        line_idx = word.get("lyrics_line_idx")
        if line_idx is None:
            if current is not None:
                records.append(current)
                current = None
                current_line_idx = None
            continue

        if line_idx != current_line_idx:
            if current is not None:
                records.append(current)
            current_line_idx = line_idx
            current = {
                "source_line_idx": line_idx,
                "text": (word.get("lyrics_line") or "").strip(),
                "word_indices": [],
            }

        current["word_indices"].append(idx)
        if not current["text"] and word.get("lyrics_line"):
            current["text"] = word.get("lyrics_line", "").strip()

    if current is not None:
        records.append(current)

    return records


def _global_align_lines(source_lines, official_lines, min_similarity=0.6):
    n = len(source_lines)
    m = len(official_lines)
    gap_penalty = -0.85
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + gap_penalty
        bt[i][0] = "up"
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + gap_penalty
        bt[0][j] = "left"

    similarities = {}
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sim = _line_similarity(source_lines[i - 1], official_lines[j - 1])
            similarities[(i - 1, j - 1)] = sim
            match_score = dp[i - 1][j - 1] + (sim * 3.0 - 1.0)
            delete_score = dp[i - 1][j] + gap_penalty
            insert_score = dp[i][j - 1] + gap_penalty
            best = max(match_score, delete_score, insert_score)
            dp[i][j] = best
            bt[i][j] = (
                "diag" if best == match_score else
                "up" if best == delete_score else
                "left"
            )

    pairs = {}
    i, j = n, m
    while i > 0 or j > 0:
        step = bt[i][j]
        if step == "diag":
            sim = similarities[(i - 1, j - 1)]
            if sim >= min_similarity:
                pairs[i - 1] = {
                    "official_idx": j - 1,
                    "similarity": sim,
                }
            i -= 1
            j -= 1
        elif step == "up":
            i -= 1
        else:
            j -= 1

    return pairs


def _align_words_in_line(source_words, official_words, min_ratio=0.58):
    source_clean = [clean_word(word) for word in source_words]
    official_clean = [clean_word(word) for word in official_words]
    n = len(source_words)
    m = len(official_words)
    gap_penalty = -0.75
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]
    ratios = {}

    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + gap_penalty
        bt[i][0] = "up"
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + gap_penalty
        bt[0][j] = "left"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            ratio = _levenshtein_ratio(source_clean[i - 1], official_clean[j - 1])
            ratios[(i - 1, j - 1)] = ratio
            match_bonus = ratio * 2.8 - 1.0
            match_score = dp[i - 1][j - 1] + match_bonus
            delete_score = dp[i - 1][j] + gap_penalty
            insert_score = dp[i][j - 1] + gap_penalty
            best = max(match_score, delete_score, insert_score)
            dp[i][j] = best
            bt[i][j] = (
                "diag" if best == match_score else
                "up" if best == delete_score else
                "left"
            )

    matches = []
    i, j = n, m
    while i > 0 or j > 0:
        step = bt[i][j]
        if step == "diag":
            ratio = ratios[(i - 1, j - 1)]
            if ratio >= min_ratio:
                matches.append((i - 1, j - 1, ratio))
            i -= 1
            j -= 1
        elif step == "up":
            i -= 1
        else:
            j -= 1

    matches.reverse()
    return matches


def remap_words_to_official_by_line(all_words, official_lyrics, min_line_similarity=0.6):
    if not all_words or not official_lyrics:
        return all_words, {
            "match_pct": 0.0,
            "matched_line_count": 0,
            "matched_word_count": 0,
        }

    line_records = _collect_line_records(all_words)
    official_lines = [line.strip() for line in official_lyrics.split("\n") if line.strip()]
    if not line_records or not official_lines:
        return all_words, {
            "match_pct": 0.0,
            "matched_line_count": 0,
            "matched_word_count": 0,
        }

    line_pairs = _global_align_lines(
        [record["text"] for record in line_records],
        official_lines,
        min_similarity=min_line_similarity,
    )

    result_words = [dict(word) for word in all_words]
    matched_word_count = 0

    for final_line_idx, record in enumerate(line_records):
        pair = line_pairs.get(final_line_idx)
        source_indices = record["word_indices"]
        source_words = [result_words[idx]["word"] for idx in source_indices]

        if not pair:
            for idx in source_indices:
                result_words[idx]["lyrics_line_idx"] = final_line_idx
                result_words[idx]["lyrics_line"] = record["text"]
            continue

        official_line = official_lines[pair["official_idx"]]
        official_words = _line_words(official_line)
        word_matches = _align_words_in_line(source_words, official_words)
        pending_skipped = []
        next_match_pointer = 0
        official_match_by_source = {src: (dst, ratio) for src, dst, ratio in word_matches}
        matched_official_indices = {dst for _, dst, _ in word_matches}

        for src_idx, word_idx in enumerate(source_indices):
            result_words[word_idx]["lyrics_line_idx"] = final_line_idx
            result_words[word_idx]["lyrics_line"] = official_line

            while next_match_pointer < len(official_words) and next_match_pointer not in matched_official_indices:
                pending_skipped.append(official_words[next_match_pointer])
                next_match_pointer += 1

            if src_idx in official_match_by_source:
                dst_idx, _ratio = official_match_by_source[src_idx]
                while next_match_pointer < dst_idx:
                    pending_skipped.append(official_words[next_match_pointer])
                    next_match_pointer += 1

                result_words[word_idx]["word"] = official_words[dst_idx]
                if pending_skipped:
                    result_words[word_idx]["skipped_before"] = pending_skipped[:]
                    pending_skipped.clear()
                matched_word_count += 1
                next_match_pointer = dst_idx + 1
            else:
                result_words[word_idx].pop("skipped_before", None)

    total_official_words = sum(len(_line_words(line)) for line in official_lines)
    match_pct = matched_word_count / max(total_official_words, 1) * 100.0
    for word in result_words:
        word["lyrics_remap_match_pct"] = match_pct
        word["lyrics_remap_mode"] = "line_by_line_experimental"

    return result_words, {
        "match_pct": match_pct,
        "matched_line_count": len(line_pairs),
        "matched_word_count": matched_word_count,
    }


def should_keep_word_token(word_text, confidence=1.0):
    clean = clean_word(word_text)
    if not clean or not re.fullmatch(r"[а-яё]+", clean):
        return False

    if len(clean) == 1:
        return clean in ALLOWED_SINGLE_CHAR_WORDS and confidence >= 0.2

    if len(clean) >= 8 and not _word_has_vowel(clean):
        return False

    return True


def extract_rhyme_words_from_lyrics(lyrics_text):
    if not lyrics_text:
        return []

    results = []
    lines = [line.strip() for line in lyrics_text.split("\n") if line.strip()]

    for idx, raw_line in enumerate(lines):
        line = _clean_lyrics_line_text(raw_line)
        words_in_line = re.findall(r"[а-яёА-ЯЁa-zA-Z]+", line)
        if not words_in_line:
            continue

        last_word = words_in_line[-1].lower().replace("ё", "е")
        if len(last_word) >= 2:
            results.append(
                {
                    "word": last_word,
                    "line": line,
                    "line_idx": idx,
                    "line_norm": clean_word(line),
                }
            )

    return results


def recover_segment_skipped_tokens(segment_text, segment_words):
    if not segment_text or not segment_words:
        return segment_words

    source_tokens = []
    for token in _line_words(segment_text):
        clean = clean_word(token)
        if not clean or not re.fullmatch(r"[а-яё]+", clean):
            continue
        source_tokens.append(token)

    if not source_tokens:
        return segment_words

    result_words = [dict(word) for word in segment_words]
    source_pointer = 0
    for word in result_words:
        target_clean = clean_word(word["word"])
        best_idx = -1
        best_ratio = 0.0
        search_end = min(len(source_tokens), source_pointer + 6)
        for idx in range(source_pointer, search_end):
            current_clean = clean_word(source_tokens[idx])
            ratio = _levenshtein_ratio(current_clean, target_clean)
            if current_clean == target_clean:
                best_idx = idx
                break
            if ratio > best_ratio and ratio >= 0.74:
                best_idx = idx
                best_ratio = ratio

        if best_idx == -1:
            continue

        skipped_before = []
        for idx in range(source_pointer, best_idx):
            if _should_preserve_skipped_token(source_tokens[idx]):
                skipped_before.append(source_tokens[idx])

        if skipped_before:
            word["skipped_before"] = skipped_before

        source_pointer = best_idx + 1

    return result_words


def get_russian_syllable_tail(word, n=3):
    word = clean_word(word)
    last_vowel_pos = -1
    for idx in range(len(word) - 1, -1, -1):
        if word[idx] in RUSSIAN_VOWELS:
            last_vowel_pos = idx
            break

    if last_vowel_pos == -1:
        return word[-n:]

    tail = word[last_vowel_pos:]
    if len(tail) < 2 and last_vowel_pos > 0:
        for idx in range(last_vowel_pos - 1, -1, -1):
            if word[idx] in RUSSIAN_VOWELS:
                tail = word[idx:]
                break
    return tail


def score_lyrics_rhyme_candidates(rhyme_words, total_lines):
    if not rhyme_words or total_lines == 0:
        return []

    word_freq_map = Counter(item["word"] for item in rhyme_words)
    line_freq_map = Counter(item["line_norm"] for item in rhyme_words)
    tail_freq_map = Counter(get_russian_syllable_tail(item["word"]) for item in rhyme_words)
    scored = []

    for entry in rhyme_words:
        clean = clean_word(entry["word"])
        progress = entry["line_idx"] / total_lines
        if progress < 0.25 or progress > 0.85:
            continue
        if len(clean) < 4 or clean in STOP_WORDS or not _word_has_vowel(clean):
            continue
        if clean in extract_separator_blocked_fragments(entry.get("line", "")):
            continue

        score = 0

        if len(clean) >= 7:
            score += 20
        elif len(clean) >= 6:
            score += SCORE_LONG_WORD_BONUS
        elif len(clean) >= 5:
            score += SCORE_MEDIUM_WORD_BONUS

        count = word_freq_map[entry["word"]]
        if count == 1:
            score += SCORE_UNIQUE_WORD_BONUS + 10
        elif count == 2:
            score += SCORE_RARE_WORD_BONUS
        elif count >= 4:
            score += SCORE_FREQUENT_PENALTY

        if clean.endswith(("ать", "ить", "ять", "еть", "ует", "ает")):
            score += SCORE_VERB_ENDING_PENALTY

        if clean.endswith(("ость", "ство", "ение", "ание")):
            score += 5

        tail = get_russian_syllable_tail(clean)
        if len(tail) >= 2 and tail_freq_map[tail] >= 2:
            score += 15
        if tail_freq_map[tail] >= 4:
            score -= 10

        if line_freq_map[entry["line_norm"]] > 1:
            score -= 25

        score += max(0, 12 - int(abs(progress - 0.62) * 40))
        scored.append((score, entry))

    scored.sort(key=lambda item: (item[0], item[1]["line_idx"]), reverse=True)
    return scored


def _occurrence_repeat_penalty(all_words, idx, answer_clean):
    penalty = 0
    left = max(0, idx - 12)
    right = min(len(all_words), idx + 5)
    repeat_count = 0
    for pointer in range(left, right):
        if pointer == idx:
            continue
        if clean_word(all_words[pointer]["word"]) == answer_clean:
            repeat_count += 1

    if repeat_count:
        penalty += 25 + max(0, repeat_count - 1) * 10

    return penalty


def _line_repeat_penalty(all_words, idx):
    lyrics_line = clean_word(all_words[idx].get("lyrics_line", ""))
    if not lyrics_line:
        return 0

    repeats = sum(1 for word in all_words if clean_word(word.get("lyrics_line", "")) == lyrics_line)
    if repeats <= 1:
        return 0
    return 20


def _score_occurrence(all_words, idx, expected_line_idx=None):
    word = all_words[idx]
    answer_clean = clean_word(word["word"])
    progress = idx / max(len(all_words), 1)
    if not (0.25 <= progress <= 0.85):
        return None
    if word["start"] <= MIN_AUDIO_POSITION:
        return None

    duration = word["end"] - word["start"]
    if duration < 0.15 or duration > 5.0:
        return None

    if idx > 0 and word["start"] < all_words[idx - 1]["end"] - 0.05:
        return None
    if idx + 1 < len(all_words) and word["end"] > all_words[idx + 1]["start"] + 0.05:
        return None
    if idx > 0 and word["start"] - all_words[idx - 1]["end"] > 5.0:
        return None

    score = 0
    if word.get("is_eol"):
        score += 20
    if expected_line_idx is not None and word.get("lyrics_line_idx") == expected_line_idx:
        score += 35
    if word.get("skipped_before"):
        score += min(8, len(word["skipped_before"]) * 2)
    if any(mark in word["word"] for mark in ",.!?"):
        score += 5

    score += min(20, int((word["start"] - MIN_AUDIO_POSITION) / 1.5))
    score += max(0, 10 - int(abs(progress - 0.6) * 30))

    score -= _occurrence_repeat_penalty(all_words, idx, answer_clean)
    score -= _line_repeat_penalty(all_words, idx)

    return score


def rank_lyrics_candidates(official_lyrics, all_words, logger=None):
    lines = [line.strip() for line in official_lyrics.split("\n") if line.strip()]
    total_lines = len(lines)
    if total_lines < 5:
        return []

    rhyme_words = extract_rhyme_words_from_lyrics(official_lyrics)
    if not rhyme_words:
        return []

    scored = score_lyrics_rhyme_candidates(rhyme_words, total_lines)
    if not scored:
        return []

    _log(logger, f"📊 Топ-5 рифм (из текста): {[(item[1]['word'], item[0]) for item in scored[:5]]}")

    ranked = []
    for lyric_score, rhyme_entry in scored[:25]:
        target_clean = clean_word(rhyme_entry["word"])
        for idx, word in enumerate(all_words):
            if clean_word(word["word"]) != target_clean:
                continue

            occurrence_score = _score_occurrence(
                all_words,
                idx,
                expected_line_idx=rhyme_entry["line_idx"],
            )
            if occurrence_score is None:
                continue

            total_score = lyric_score + occurrence_score
            ranked.append(
                {
                    "idx": idx,
                    "line": rhyme_entry["line"],
                    "line_idx": rhyme_entry["line_idx"],
                    "score": total_score,
                    "word": rhyme_entry["word"],
                }
            )

    if not ranked:
        _log(logger, "⚠️ Ни один кандидат из текста не найден в массиве слов")
        return []

    ranked.sort(key=lambda item: (item["score"], item["idx"]), reverse=True)
    return ranked


def select_word_from_lyrics_algorithmically(official_lyrics, all_words, logger=None):
    ranked = rank_lyrics_candidates(official_lyrics, all_words, logger=logger)
    if not ranked:
        return -1, "", None
    best = ranked[0]
    _log(logger, f"✅ [LYRICS_ALGO] Выбрано: '{best['word']}' (score={best['score']})")
    return best["idx"], best["line"], best.get("line_idx")


def score_candidates(all_words, extended_range=False, logger=None):
    if not all_words:
        return []

    scores = []
    total_words = len(all_words)
    freq_map = Counter(clean_word(word["word"]) for word in all_words)

    min_progress = 0.20 if extended_range else 0.30
    max_progress = 0.85 if extended_range else 0.72
    min_audio_pos = MIN_AUDIO_POSITION * 0.7 if extended_range else MIN_AUDIO_POSITION

    eol_tails = {}
    for idx, word in enumerate(all_words):
        if word.get("is_eol"):
            clean = clean_word(word["word"])
            if len(clean) >= 3:
                eol_tails[idx] = get_russian_syllable_tail(clean)

    for idx, word in enumerate(all_words):
        clean = clean_word(word["word"])
        if len(clean) < 4 or clean in STOP_WORDS:
            continue
        if is_separator_fragment_candidate(word):
            continue

        progress = idx / total_words
        if progress < min_progress or progress > max_progress:
            continue
        if word["start"] < min_audio_pos:
            continue

        duration = word["end"] - word["start"]
        if duration < 0.15 or duration > 5.0:
            continue
        if idx > 0 and word["start"] < all_words[idx - 1]["end"] - 0.05:
            continue
        if idx + 1 < len(all_words) and word["end"] > all_words[idx + 1]["start"] + 0.05:
            continue

        score = 0
        if word.get("is_eol"):
            score += SCORE_EOL_BONUS
        if any(mark in word["word"] for mark in ",.!?"):
            score += SCORE_PUNCTUATION_BONUS

        if len(clean) >= 6:
            score += SCORE_LONG_WORD_BONUS
        elif len(clean) >= 5:
            score += SCORE_MEDIUM_WORD_BONUS

        count = freq_map[clean]
        if count == 1:
            score += SCORE_UNIQUE_WORD_BONUS
        elif count == 2:
            score += SCORE_RARE_WORD_BONUS
        elif count >= 4:
            score += SCORE_FREQUENT_PENALTY

        if clean.endswith(("ать", "ить", "ять", "еть", "ует", "ает")):
            score += SCORE_VERB_ENDING_PENALTY

        if word.get("is_eol") and len(clean) >= 3:
            my_tail = get_russian_syllable_tail(clean)
            for other_idx, other_tail in eol_tails.items():
                if other_idx != idx and other_tail == my_tail:
                    score += 15
                    break

        if idx > 0:
            gap = word["start"] - all_words[idx - 1]["end"]
            if gap > 2.0:
                score -= 20
            elif gap < 0.01:
                score -= 15

        if word["start"] > 120.0:
            score -= 15
        elif word["start"] > 80.0:
            score -= 8

        score += max(0, 10 - int(abs(progress - 0.58) * 30))
        scores.append((score, idx))

    scores.sort(key=lambda item: (item[0], item[1]), reverse=True)
    top_debug = [(all_words[idx]["word"], score) for score, idx in scores[:5]]
    _log(logger, f"📊 Топ-5 кандидатов (алгоритм): {top_debug}")
    return [idx for score, idx in scores]


def get_algorithmic_choice(all_words, extended_range=False, logger=None):
    ranked_indices = score_candidates(
        all_words,
        extended_range=extended_range,
        logger=logger,
    )
    if not ranked_indices:
        return None
    return ranked_indices[0]


def calculate_timings(all_words, target_idx, is_lyrics_mode=False, answer_line=""):
    if not all_words or target_idx < 0 or target_idx >= len(all_words):
        return (0, 0), (0, 0)

    target_word = all_words[target_idx]
    target_start_ms = int(target_word["start"] * 1000)
    cut_ms = max(0, target_start_ms - QUESTION_WORD_GUARD_MS)
    raw_start_ms = max(0, cut_ms - TARGET_QUESTION_DURATION_MS)

    q_start_ms = raw_start_ms
    best_diff = float("inf")

    for word in all_words:
        word_start_ms = int(word["start"] * 1000)
        duration_candidate = cut_ms - word_start_ms
        if duration_candidate < MIN_QUESTION_DURATION_MS:
            break

        diff = abs(word_start_ms - raw_start_ms)
        if diff < 3000 and diff < best_diff:
            best_diff = diff
            q_start_ms = max(0, word_start_ms - 100)

    q_end_ms = cut_ms
    if (q_end_ms - q_start_ms) < MIN_QUESTION_DURATION_MS:
        q_start_ms = max(0, cut_ms - 26000)

    a_start_ms = max(0, target_start_ms - 5000)
    a_duration_ms = 15000
    a_end_ms = min(int(all_words[-1]["end"] * 1000) + 1000, a_start_ms + a_duration_ms)

    return (q_start_ms, q_end_ms), (a_start_ms, a_end_ms)


def question_window_has_enough_context(
    all_words,
    target_idx,
    q_start_ms,
    q_end_ms,
    *,
    min_words_before_target=5,
    min_meaningful_words=3,
    max_initial_silence_ms=3500,
):
    if not all_words or target_idx <= 0:
        return False, "too_few_words"

    window_words = []
    for idx, word in enumerate(all_words):
        start_ms = int(word["start"] * 1000)
        if q_start_ms <= start_ms < q_end_ms:
            window_words.append((idx, word))

    pre_answer_words = [item for item in window_words if item[0] < target_idx]
    if pre_answer_words:
        first_word_start_ms = int(pre_answer_words[0][1]["start"] * 1000)
        if first_word_start_ms - q_start_ms > max_initial_silence_ms:
            return False, "leading_instrumental"

    if len(pre_answer_words) < min_words_before_target:
        return False, "too_few_words"

    meaningful_words = [
        clean_word(word["word"])
        for _, word in pre_answer_words
        if clean_word(word["word"]) not in STOP_WORDS and len(clean_word(word["word"])) >= 3
    ]
    if len(meaningful_words) < min_meaningful_words:
        return False, "too_few_meaningful_words"

    return True, "ok"


def _build_context_from_answer_line(target_word, answer_line):
    tokens = _visible_text_tokens(answer_line)
    if not tokens:
        return ""

    answer_clean = clean_word(target_word.get("word", ""))
    if not answer_clean:
        return ""

    best_idx = -1
    best_ratio = 0.0
    for idx, token in enumerate(tokens):
        token_clean = clean_word(token)
        if not token_clean:
            continue
        if token_clean == answer_clean:
            best_idx = idx
            best_ratio = 1.0
        else:
            ratio = _levenshtein_ratio(token_clean, answer_clean)
            if ratio >= 0.74 and ratio >= best_ratio:
                best_idx = idx
                best_ratio = ratio

    if best_idx <= 0:
        return ""

    return " ".join(tokens[:best_idx]).strip()


def _find_previous_lyrics_line(all_words, target_idx):
    if not (0 <= target_idx < len(all_words)):
        return ""

    target_line_idx = all_words[target_idx].get("lyrics_line_idx")
    if target_line_idx is None:
        return ""

    best_line_idx = None
    best_line_text = ""
    for idx in range(target_idx - 1, -1, -1):
        line_idx = all_words[idx].get("lyrics_line_idx")
        line_text = _clean_lyrics_line_text(all_words[idx].get("lyrics_line", ""))
        if line_idx is None or not line_text:
            continue
        if line_idx >= target_line_idx:
            continue
        if best_line_idx is None or line_idx > best_line_idx:
            best_line_idx = line_idx
            best_line_text = line_text
            if line_idx == target_line_idx - 1:
                break

    return best_line_text


def _get_previous_official_lyrics_line(lyrics_text, answer_line_idx):
    if not lyrics_text or answer_line_idx is None or answer_line_idx <= 0:
        return ""

    lines = [_clean_lyrics_line_text(line) for line in lyrics_text.split("\n") if line.strip()]
    if answer_line_idx >= len(lines):
        return ""

    return lines[answer_line_idx - 1]


def _should_break_general_context_line(previous_word, current_word):
    if not previous_word or not current_word:
        return False

    if previous_word.get("is_eol"):
        return True

    gap = current_word["start"] - previous_word["end"]
    return gap > 0.55


def build_context_string(all_words, target_idx, **kwargs):
    is_lyrics_mode = kwargs.get("is_lyrics_mode", False)
    answer_line = kwargs.get("answer_line", "")
    answer_line_idx = kwargs.get("answer_line_idx")
    lyrics_text = kwargs.get("lyrics_text", "")
    if is_lyrics_mode and answer_line and 0 <= target_idx < len(all_words):
        line_context = _build_context_from_answer_line(all_words[target_idx], answer_line)
        if line_context:
            previous_line = _get_previous_official_lyrics_line(lyrics_text, answer_line_idx)
            if not previous_line:
                previous_line = _find_previous_lyrics_line(all_words, target_idx)
            context_parts = []
            if previous_line and clean_word(previous_line) != clean_word(line_context):
                context_parts.append(previous_line)
            context_parts.append(line_context)
            return "\n".join(context_parts) + " ___"

    def _append_without_overlap(target_tokens, extra_tokens):
        if not extra_tokens:
            return

        clean_target = [clean_word(token) for token in target_tokens]
        clean_extra = [clean_word(token) for token in extra_tokens]
        if not clean_extra:
            return

        for start in range(0, max(0, len(clean_target) - len(clean_extra) + 1)):
            if clean_target[start:start + len(clean_extra)] == clean_extra:
                return

        max_overlap = min(len(clean_target), len(clean_extra))
        best_start = 0
        best_size = 0

        for size in range(max_overlap, 0, -1):
            suffix = clean_target[-size:]
            max_start = len(clean_extra) - size
            for start in range(0, max_start + 1):
                if suffix == clean_extra[start:start + size]:
                    best_start = start
                    best_size = size
                    break
            if best_size:
                break

        target_tokens.extend(extra_tokens[best_start + best_size:])

    start_idx = max(0, target_idx - 12)
    context_indices = []
    for idx in range(target_idx - 1, start_idx - 1, -1):
        current_word = all_words[idx]
        if idx > 0:
            previous_word = all_words[idx - 1]
            if (current_word["start"] - previous_word["end"]) > 1.2:
                context_indices.insert(0, idx)
                break
        context_indices.insert(0, idx)

    if len(context_indices) < 2:
        start_idx = max(0, target_idx - 8)
        context_indices = list(range(start_idx, target_idx))

    context_lines = []
    current_line_words = []
    previous_context_word = None
    for idx in context_indices:
        current_word = all_words[idx]
        if current_line_words and _should_break_general_context_line(previous_context_word, current_word):
            context_lines.append(current_line_words)
            current_line_words = []

        skipped_before = all_words[idx].get("skipped_before", [])
        if skipped_before:
            _append_without_overlap(current_line_words, skipped_before)
        _append_without_overlap(current_line_words, [all_words[idx]["word"]])
        previous_context_word = current_word

    if current_line_words:
        context_lines.append(current_line_words)

    target_word = all_words[target_idx]
    skipped = target_word.get("skipped_before", [])
    if skipped:
        if not context_lines:
            context_lines = [[]]
        _append_without_overlap(context_lines[-1], skipped)
    elif target_idx > 0:
        prev_end = all_words[target_idx - 1]["end"]
        gap = target_word["start"] - prev_end
        if gap > 0.8:
            if not context_lines:
                context_lines = [[]]
            context_lines[-1].append("...")

    if len(context_lines) >= 2:
        visible_lines = context_lines[-2:]
        context = "\n".join(" ".join(tokens).strip() for tokens in visible_lines if tokens)
    else:
        flat_tokens = context_lines[0] if context_lines else []
        context = " ".join(flat_tokens)

    return context.strip() + " ___"


def build_question_payload(question_id, context_str, answer_text, track_meta="", has_lyrics=False):
    payload = {
        "id": question_id,
        "type": "text",
        "question": context_str,
        "answer": answer_text,
        "options": [],
        "track_meta": "",
    }
    if track_meta and not has_lyrics:
        payload["track_meta"] = track_meta
    return payload


qlogic = SimpleNamespace(
    QUESTION_WORD_GUARD_MS=QUESTION_WORD_GUARD_MS,
    build_context_string=build_context_string,
    calculate_timings=calculate_timings,
    get_algorithmic_choice=get_algorithmic_choice,
    question_window_has_enough_context=question_window_has_enough_context,
    rank_lyrics_candidates=rank_lyrics_candidates,
    recover_segment_skipped_tokens=recover_segment_skipped_tokens,
    score_candidates=score_candidates,
    select_word_from_lyrics_algorithmically=select_word_from_lyrics_algorithmically,
    should_keep_word_token=should_keep_word_token,
)


def generation_task(game_id, token, urls):
    global _gen_start_time
    _gen_start_time = time.time()

    with job_status_lock:
        job_status["is_busy"] = True
        job_status["progress"] = 0
        job_status["logs"] = []
        job_status["status"] = "running"

    # Статистика генерации
    generation_stats = {
        'total_tracks': 0,
        'tracks_processed': 0,
        'lyrics_algo_success': 0,    # Алгоритмический выбор рифмы из текста (Whisper таймкоды)
        'algo_success': 0,
        'skipped_short': 0,
        'skipped_no_context': 0,
        'questions_created': 0
    }

    game_media_folder = os.path.join(MEDIA_ROOT, f"{game_id}-media")
    game_temp_folder = os.path.join(MEDIA_ROOT, f"{game_id}-temp_downloads")
    game_json_file = os.path.join(MEDIA_ROOT, f"{game_id}-questions.json")
    
    if not os.path.exists(game_media_folder): os.makedirs(game_media_folder)
    if not os.path.exists(game_temp_folder): os.makedirs(game_temp_folder)
    
    try:
        urls, duplicate_refs = dedupe_yandex_track_urls(urls)
        if duplicate_refs:
            log(f"🧹 Убраны дубли ссылок перед скачиванием: {len(duplicate_refs)}")
            for duplicate_ref in duplicate_refs:
                log(f"   ↩️ duplicate: {duplicate_ref}")

        client = None
        yandex_throttle = YandexRequestThrottle(
            min_interval_seconds=YANDEX_REQUEST_MIN_INTERVAL_SECONDS
        )
        if token:
            try:
                client = YandexClient(token)
                log("✅ Яндекс авторизован")
            except Exception as e:
                log(f"⚠️ Ошибка авторизации Яндекс: {e}")

        downloaded = []
        # Храним метаданные и ТЕКСТЫ
        files_metadata = {} 

        for url in urls:
            try:
                match = re.search(r'track/(\d+)', url)
                if match and client:
                    tid = match.group(1)
                    fpath = os.path.join(game_temp_folder, f"{tid}.mp3")
                    track_info = None

                    if not os.path.exists(fpath):
                        for attempt in range(3):
                            try:
                                yandex_throttle.wait()
                                track_info = client.tracks([tid])[0]
                                track_info.download(fpath)
                                break
                            except Exception as dl_e:
                                if attempt < 2:
                                    yandex_throttle.penalize(YANDEX_REQUEST_FAILURE_COOLDOWN_SECONDS)
                                    log(f"⚠️ Попытка {attempt+1}/3 не удалась для {tid}: {dl_e}. Повтор через {2 ** attempt}с...")
                                    time.sleep(2 ** attempt)
                                else:
                                    raise dl_e
                    downloaded.append(fpath)
                    
                    # --- ПОЛУЧАЕМ ИНФОРМАЦИЮ О ТРЕКЕ ---
                    try:
                        if track_info is None:
                            for attempt in range(3):
                                try:
                                    yandex_throttle.wait()
                                    track_info = client.tracks([tid])[0]
                                    break
                                except Exception as ti_e:
                                    if attempt < 2:
                                        yandex_throttle.penalize(YANDEX_REQUEST_FAILURE_COOLDOWN_SECONDS)
                                        log(f"⚠️ Метаданные {tid}: попытка {attempt+1}/3 — {ti_e}. Повтор через {2 ** attempt}с...")
                                        time.sleep(2 ** attempt)
                                    else:
                                        raise ti_e
                        artist = track_info.artists[0].name if track_info.artists else "Неизвестен"
                        title = track_info.title
                        
                        # 1. Пробуем достать официальный текст
                        lyrics = ""
                        raw_lrc = ""  # Сырой LRC с таймкодами
                        lyrics_source = "none"

                        # Способ 1: Новый API (get_lyrics + fetch_lyrics) — 2025-2026
                        for attempt in range(3):
                            try:
                                yandex_throttle.wait()
                                lyrics_obj = track_info.get_lyrics('LRC')
                                if lyrics_obj:
                                    yandex_throttle.wait()
                                    raw_lrc = lyrics_obj.fetch_lyrics()
                                    lyrics = clean_lrc_lyrics(raw_lrc)
                                    lyrics_source = "lrc"
                                    log(f"📜 Найден текст (LRC) для: {title}")
                                break  # Успех или текста нет — не повторяем
                            except NotFoundError:
                                break  # Текст не найден — повторять бессмысленно
                            except Exception as lrc_e:
                                if attempt < 2:
                                    yandex_throttle.penalize(YANDEX_REQUEST_FAILURE_COOLDOWN_SECONDS)
                                    log(f"⚠️ LRC {tid}: попытка {attempt+1}/3 — {lrc_e}. Повтор через {2 ** attempt}с...")
                                    time.sleep(2 ** attempt)
                                else:
                                    log(f"⚠️ LRC {tid}: все 3 попытки неудачны, пробуем supplement")

                        # Способ 2: Fallback на старый API (supplement)
                        if not lyrics:
                            for attempt in range(3):
                                try:
                                    yandex_throttle.wait()
                                    supp = track_info.get_supplement()
                                    if supp and supp.lyrics and supp.lyrics.full_lyrics:
                                        lyrics = supp.lyrics.full_lyrics
                                        lyrics_source = "supplement"
                                        log(f"📜 Найден текст (supplement) для: {title}")
                                    break
                                except Exception as sup_e:
                                    if attempt < 2:
                                        yandex_throttle.penalize(YANDEX_REQUEST_FAILURE_COOLDOWN_SECONDS)
                                        log(f"⚠️ Supplement {tid}: попытка {attempt+1}/3 — {sup_e}. Повтор через {2 ** attempt}с...")
                                        time.sleep(2 ** attempt)
                                    else:
                                        log(f"⚠️ Supplement {tid}: текст не получен после 3 попыток")

                        # Сохраняем всё в словарь
                        files_metadata[fpath] = {
                            "meta": f"{artist} - {title}",
                            "lyrics": lyrics,
                            "raw_lrc": raw_lrc,
                            "lyrics_source": lyrics_source
                        }
                        
                        log(f"📥 Скачан: {tid} ({artist} - {title})")
                    except Exception as meta_e:
                        log(f"📥 Скачан: {tid} (без метаданных)")
                        files_metadata[fpath] = {"meta": "", "lyrics": "", "raw_lrc": "", "lyrics_source": "none"}
                        
            except Exception as e:
                log(f"⚠️ Ошибка URL {url}: {e}")

        if not downloaded:
            log("❌ Нет файлов")
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"
        new_qs = []
        total = len(downloaded)
        generation_stats['total_tracks'] = total

        for idx, fpath in enumerate(downloaded):
            generation_stats['tracks_processed'] += 1
            # Лог использования RAM для диагностики утечек памяти
            try:
                mem_mb = psutil.Process().memory_info().rss // 1024 // 1024
                log(f"⚙️ Трек {idx+1}/{total} | RAM: {mem_mb}MB")
            except:
                log(f"⚙️ Трек {idx+1}/{total}")
            
            # Очистка временных папок demucs (если использовались ранее)
            try:
                shutil.rmtree(os.path.join(os.path.dirname(fpath), "htdemucs"), ignore_errors=True)
                shutil.rmtree(os.path.join(os.path.dirname(fpath), "htdemucs_ft"), ignore_errors=True)
            except OSError:
                pass  # Не критично — папки могут не существовать

            try:
                # Достаем инфу
                file_data = files_metadata.get(fpath, {"meta": "", "lyrics": "", "raw_lrc": "", "lyrics_source": "none"})
                has_lyrics = bool(file_data["lyrics"] and len(file_data["lyrics"]) > 50)
                answer_line = ""  # Строка текста с ответом (для lyrics-режима)
                answer_line_idx = None

                # ===== ПАЙПЛАЙН: Roformer MDX → Whisper =====
                track_t0 = time.time()

                vocals_path = isolate_vocals(fpath)
                recognition_source = preprocess_for_whisper(vocals_path)

                words = process_audio_with_whisperx(
                    recognition_source,
                    device=device,
                    song_meta=file_data["meta"],
                    official_lyrics=file_data["lyrics"]
                )

                track_elapsed = time.time() - track_t0
                log(f"⏱️ MDX+Whisper: {track_elapsed:.1f}с | {len(words)} слов")

                if len(words) < 15:
                    log("🚫 Мало слов от Whisper. Пропуск.")
                    generation_stats['skipped_short'] += 1
                    continue

                # 4. ВЫБОР СЛОВА: только алгоритмический пайплайн, без LLM
                if has_lyrics:
                    log(f"📜 Режим: WHISPER + LYRICS ALGO (текст из {file_data['lyrics_source']})")

                    # 4a. Совмещаем Whisper-таймкоды с точным текстом из lyrics
                    words = align_words_to_lyrics(words, file_data["lyrics"])
                    remapped_words, remap_stats = remap_words_to_official_by_line(words, file_data["lyrics"])
                    if remap_stats["matched_line_count"] > 0:
                        words = remapped_words
                        log(
                            "🔁 Line remap: "
                            f"{remap_stats['matched_line_count']} lines, "
                            f"{remap_stats['match_pct']:.1f}% words"
                        )

                    # 4b. Алгоритмический выбор рифмы из текста, таймкоды из Whisper
                    target_idx, answer_line, answer_line_idx = qlogic.select_word_from_lyrics_algorithmically(
                        file_data["lyrics"],
                        words,
                        logger=log,
                    )
                    used_method = "lyrics_algo"

                    if target_idx == -1:
                        log("⚠️ Lyrics algo не нашёл кандидата, fallback на детерминированный общий алгоритм")
                        target_idx = qlogic.get_algorithmic_choice(words, logger=log)
                        used_method = "algo"
                    else:
                        generation_stats['lyrics_algo_success'] += 1

                else:
                    log(f"🎤 Режим: WHISPER + ALGO (текст не найден)")

                    target_idx = qlogic.get_algorithmic_choice(words, logger=log)
                    used_method = "algo"

                if target_idx == -1 or target_idx is None:
                    log("❌ Не удалось выбрать слово.")
                    continue

                if used_method == "algo":
                    generation_stats['algo_success'] += 1

                # Пробуем найти слово с достаточным контекстом (до 5 попыток)
                tried_indices = set()
                final_target_idx = None
                lyrics_ranked_candidates = None

                for timing_attempt in range(5):
                    if target_idx in tried_indices:
                        if has_lyrics:
                            if lyrics_ranked_candidates is None:
                                lyrics_ranked_candidates = qlogic.rank_lyrics_candidates(
                                    file_data["lyrics"],
                                    words,
                                    logger=log,
                                )
                            ranked = lyrics_ranked_candidates
                        else:
                            # После 2 попыток расширяем диапазон поиска
                            extended = timing_attempt >= 2
                            ranked = qlogic.score_candidates(words, extended_range=extended, logger=log)
                        found_new = False
                        for candidate_idx in ranked:
                            if has_lyrics:
                                candidate_id = candidate_idx["idx"]
                            else:
                                candidate_id = candidate_idx

                            if candidate_id not in tried_indices:
                                target_idx = candidate_id
                                if has_lyrics:
                                    answer_line = candidate_idx["line"]
                                    answer_line_idx = candidate_idx.get("line_idx")
                                found_new = True
                                break
                        if not found_new:
                            break  # Все варианты исчерпаны

                    tried_indices.add(target_idx)
                    q_times, a_times = qlogic.calculate_timings(
                        words,
                        target_idx,
                        is_lyrics_mode=has_lyrics,
                        answer_line=answer_line,
                    )

                    # Проверка 1: достаточная длительность контекста
                    if (q_times[1] - q_times[0]) < 20000:
                        log(f"⚠️ Контекст короткий для '{words[target_idx]['word']}', пробуем другое слово... (попытка {timing_attempt + 1}/5)")
                        continue

                    has_context, context_reason = qlogic.question_window_has_enough_context(
                        words,
                        target_idx,
                        q_times[0],
                        q_times[1],
                    )
                    if not has_context:
                        log(
                            f"⚠️ Контекст-проигрыш для '{words[target_idx]['word']}' "
                            f"({context_reason}), пробуем другое... (попытка {timing_attempt + 1}/5)"
                        )
                        continue

                    # Проверка 2: слово-ответ не должно звучать в аудио вопроса (spoiler)
                    if check_spoiler_in_question(words, target_idx, q_times[0], q_times[1]):
                        log(f"⚠️ Spoiler: '{words[target_idx]['word']}' звучит в аудио вопроса, пробуем другое... (попытка {timing_attempt + 1}/5)")
                        continue

                    # Проверка 3: пауза перед ответом не слишком большая (проигрыш)
                    if target_idx > 0:
                        gap_before = words[target_idx]['start'] - words[target_idx - 1]['end']
                        if gap_before > 5.0:
                            log(f"⚠️ Большая пауза ({gap_before:.1f}с) перед '{words[target_idx]['word']}', пробуем другое... (попытка {timing_attempt + 1}/5)")
                            continue

                    # Проверка 4: таймкоды не сжаты (слово должно длиться >0.15с)
                    word_duration = words[target_idx]['end'] - words[target_idx]['start']
                    if word_duration < 0.15:
                        log(f"⚠️ Сжатый таймкод ({word_duration:.2f}с) для '{words[target_idx]['word']}', пробуем другое... (попытка {timing_attempt + 1}/5)")
                        continue

                    # Проверка 5: таймкод не раздут (>5с на одно слово = баг alignment)
                    if word_duration > 5.0:
                        log(f"⚠️ Раздутый таймкод ({word_duration:.1f}с) для '{words[target_idx]['word']}', пробуем другое... (попытка {timing_attempt + 1}/5)")
                        continue

                    # Проверка 6: нет перекрытия с соседними словами
                    if target_idx > 0 and words[target_idx]['start'] < words[target_idx - 1]['end'] - 0.05:
                        log(f"⚠️ Перекрытие с предыдущим словом для '{words[target_idx]['word']}', пробуем другое... (попытка {timing_attempt + 1}/5)")
                        continue

                    # Все проверки пройдены
                    final_target_idx = target_idx
                    break

                if final_target_idx is None:
                    log(f"❌ Не найдено слово с достаточным контекстом. Пропуск.")
                    generation_stats['skipped_no_context'] += 1
                    continue

                target_idx = final_target_idx
                answer_word_obj = words[target_idx]
                answer_text = re.sub(r'[^\w]', '', answer_word_obj['word'])

                # --- DEBUG: таймкоды вокруг слова-ответа ---
                dbg_start = max(0, target_idx - 5)
                dbg_end = min(len(words), target_idx + 3)
                dbg_lines = []
                for di in range(dbg_start, dbg_end):
                    w = words[di]
                    marker = " <<<" if di == target_idx else ""
                    dbg_lines.append(f"  [{di}] {w['start']:.2f}-{w['end']:.2f}s '{w['word']}'{marker}")
                log(f"🔍 Таймкоды (q_end={q_times[1]}ms, a_start={a_times[0]}ms):\n" + "\n".join(dbg_lines))

                audio = AudioSegment.from_mp3(fpath)
                orig_duration_ms = len(audio)
                q_seg = audio[q_times[0]:q_times[1]].fade_out(120).fade_in(1500)
                a_seg = audio[a_times[0]:a_times[1]].fade_in(100)
                log(f"🔍 Оригинал: {orig_duration_ms}ms, q=[{q_times[0]}-{q_times[1]}ms] ({len(q_seg)}ms), a=[{a_times[0]}-{a_times[1]}ms] ({len(a_seg)}ms)")
                
                qid = str(uuid.uuid4())[:8]
                q_seg.export(os.path.join(game_media_folder, f"{qid}-1.mp3"), format="mp3")
                a_seg.export(os.path.join(game_media_folder, f"{qid}-2.mp3"), format="mp3")
                
                context_str = qlogic.build_context_string(
                    words,
                    target_idx,
                    is_lyrics_mode=has_lyrics,
                    answer_line=answer_line,
                    answer_line_idx=answer_line_idx,
                    lyrics_text=file_data["lyrics"],
                )
                
                new_qs.append(
                    build_question_payload(
                        question_id=qid,
                        context_str=context_str,
                        answer_text=answer_text,
                        track_meta=file_data.get("meta", ""),
                        has_lyrics=has_lyrics,
                    )
                )

                generation_stats['questions_created'] += 1
                log(f"✅ [{used_method.upper()}] Ответ: {answer_text}")
                
            except Exception as e:
                log(f"❌ Ошибка трека: {e}")
                traceback.print_exc()
            
            job_status["progress"] = int(((idx + 1) / total) * 100)
            gc.collect()
            # Принудительно освобождаем память на уровне libc (Linux)
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except:
                pass  # Windows/macOS не поддерживают

        with open(game_json_file, 'w', encoding='utf-8') as f:
            json.dump(new_qs, f, ensure_ascii=False, indent=4)

        try:
            shutil.rmtree(game_temp_folder)
            shutil.rmtree(os.path.join(game_temp_folder, "..", "htdemucs_ft"), ignore_errors=True)
            shutil.rmtree(os.path.join(game_temp_folder, "..", "htdemucs"), ignore_errors=True)
        except OSError:
            pass  # Очистка не удалась — не критично

        # Детальная статистика
        log(f"\n📊 Статистика генерации:")
        log(f"   Всего треков: {generation_stats['total_tracks']}")
        log(f"   Вопросов создано: {generation_stats['questions_created']}")
        if generation_stats['lyrics_algo_success'] > 0:
            log(f"   Lyrics Algo (Whisper + текст): {generation_stats['lyrics_algo_success']}")
        if generation_stats['algo_success'] > 0:
            log(f"   General Algo: {generation_stats['algo_success']}")
        if generation_stats['skipped_short'] > 0:
            log(f"   Пропущено (мало слов): {generation_stats['skipped_short']}")
        if generation_stats['skipped_no_context'] > 0:
            log(f"   Пропущено (короткий контекст): {generation_stats['skipped_no_context']}")

    except Exception as e:
        log(f"FATAL: {e}")
        logger.exception("Generation task failed")
    finally:
        # Освобождаем кэшированную модель Separator
        release_separator()
        with job_status_lock:
            job_status["is_busy"] = False
            job_status["status"] = "finished"

def process_audio_with_whisperx(audio_path, device="cpu", song_meta="", official_lyrics=""):
    """
    Распознает текст, используя WhisperX.
    official_lyrics: если есть, добавляется в промпт для идеальной точности.
    """
    try:
        log(f"loading model {WHISPER_SIZE} on {device}...")
        tuning = get_runtime_tuning(device)

        music_vad_options = {
            "vad_onset": 0.1,             
            "vad_offset": 0.36,            
            "min_speech_duration_ms": 200, 
        }

        # --- ФОРМИРОВАНИЕ УМНОГО ПРОМПТА ---
        base_prompt = "Текст песни на русском языке. Рифма, куплеты, припев."
        
        prompt = base_prompt
        # 1. Добавляем метаданные (Воровайки - Хоп мусорок)
        if song_meta:
            prompt += f" Песня: {song_meta}."
            
        # 2. Добавляем официальный текст (ЧИТ-КОД!)
        # Берем первые 200 символов, чтобы Whisper понял стиль и орфографию,
        # но не переполнил контекстное окно.
        if official_lyrics:
            clean_lyrics_preview = official_lyrics.replace('\n', ' ')[:220]
            prompt += f" Слова: {clean_lyrics_preview}..."
            log("📜 Prompt Lyrics injected")
            log(clean_lyrics_preview)

        asr_options = {
            "beam_size": 10,
            "best_of": 5,
            "patience": 2.0,
            "length_penalty": 1.0,

            # Параметры ASR — оптимизированы для ПЕНИЯ (не речи!)
            "log_prob_threshold": -1.0,
            "no_speech_threshold": 0.4,    # Снижено с 0.8 — в пении голос часто похож на "не речь"

            "initial_prompt": prompt,
            "hallucination_silence_threshold": 4.0,  # Повышено с 2.0 — в песнях паузы между фразами >2с это норма
            "condition_on_previous_text": True,
            "suppress_blank": True,
            "word_timestamps": True
        }

        # 8845HS: 8 ядер / 16 потоков, 29GB RAM — можно увеличить параллелизм
        model = whisperx.load_model(
            WHISPER_SIZE,
            device,
            compute_type=COMPUTE_TYPE,
            language="ru",
            vad_options=music_vad_options,
            asr_options=asr_options,
            threads=tuning["whisper_threads"]
        )

        audio = whisperx.load_audio(audio_path)

        result = model.transcribe(
            audio,
            batch_size=tuning["whisper_batch_size"],
            language="ru",
            task="transcribe"
        )
        
        raw_segments = result["segments"]

        # Фильтруем мусорные сегменты ДО alignment (иначе alignment падает с WARNING)
        filtered_segments = []
        for seg in raw_segments:
            seg_text = seg.get("text", "")
            if is_junk_segment(seg_text):
                log(f"🗑️ Мусор до alignment: '{seg_text}'")
                continue
            filtered_segments.append(seg)

        del model
        gc.collect()
        if device == "cuda": torch.cuda.empty_cache()

        log("Aligning audio...")
        model_a = None
        metadata_a = None
        try:
            model_a, metadata_a = whisperx.load_align_model(language_code=result["language"], device=device)
            aligned_result = whisperx.align(filtered_segments, model_a, metadata_a, audio, device, return_char_alignments=False)
            segments_to_process = aligned_result["segments"]
        except Exception as e:
            log(f"⚠️ Ошибка Alignment: {e}. Используем сырые сегменты.")
            segments_to_process = filtered_segments
        finally:
            # Гарантированная очистка модели и метаданных
            if model_a is not None:
                del model_a
            if metadata_a is not None:
                del metadata_a
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

        # Освобождаем аудио-буфер после завершения всех операций с ним
        del audio
        gc.collect()

        all_words = []
        for segment in segments_to_process:
            seg_text = segment.get("text", "")
            if is_junk_segment(seg_text): continue

            words_in_seg = segment.get("words", [])
            segment_word_objects = []
            
            if not words_in_seg and "start" in segment and "end" in segment:
                log("⚠️ Fallback: используем сегмент целиком")
                all_words.append({
                    "word": segment["text"].strip(),
                    "start": segment["start"],
                    "end": segment["end"],
                    "is_eol": True,
                    "confidence": 0.5,
                    "segment_text": seg_text,
                })
                continue

            for i, word in enumerate(words_in_seg):
                if "start" not in word or "end" not in word: continue

                confidence = word.get("score", 1.0)
                min_confidence = 0.2 if len(clean_word(word["word"])) == 1 else 0.4
                if confidence < min_confidence:
                    continue

                duration = word["end"] - word["start"]
                if duration < 0.05 or duration > 4.0: continue

                if not qlogic.should_keep_word_token(word["word"], confidence=confidence):
                    continue

                # Определяем конец строки по паузе после слова
                is_end_of_line = (i == len(words_in_seg) - 1)  # Конец сегмента
                if not is_end_of_line and i < len(words_in_seg) - 1:
                    next_word = words_in_seg[i + 1]
                    if "start" in next_word:
                        pause_after = next_word["start"] - word["end"]
                        # Пауза > 0.3 сек = вероятный конец строки (рифма)
                        if pause_after > 0.3:
                            is_end_of_line = True

                w_obj = {
                    "word": word["word"].strip(),
                    "start": word["start"],
                    "end": word["end"],
                    "is_eol": is_end_of_line,
                    "confidence": confidence,
                    "segment_text": seg_text,
                }
                segment_word_objects.append(w_obj)

            if segment_word_objects:
                recovered_words = qlogic.recover_segment_skipped_tokens(seg_text, segment_word_objects)
                all_words.extend(recovered_words)

        # --- ДЕБАГ: выводим распознанный текст ---
        if all_words:
            full_text_parts = []
            for w in all_words:
                full_text_parts.append(w['word'])
                if w.get('is_eol'):
                    full_text_parts.append('\n')
            full_text = ' '.join(full_text_parts).replace(' \n ', '\n').replace(' \n', '\n')
            log(f"📝 [WHISPER] Текст ({len(all_words)} слов):\n{full_text}")

        return all_words
    except Exception as e:
        log(f"WhisperX Error: {e}")
        traceback.print_exc()
        return []

# --- ИНТЕЛЛЕКТУАЛЬНЫЙ ВЫБОР ---

# Константы MIN_AUDIO_POSITION уже определены выше в секции "Scoring"
# Здесь только утилиты и функции

def check_spoiler_in_question(all_words, target_idx, q_start_ms, q_end_ms):
    """
    Проверяет, не звучит ли слово-ответ в аудио вопроса (spoiler).
    Возвращает True если есть spoiler, False если всё ок.
    """
    if target_idx < 0 or target_idx >= len(all_words):
        return True  # Некорректный индекс = spoiler

    answer_clean = clean_word(all_words[target_idx]['word'])

    # Проверяем все вхождения этого слова
    for i, w in enumerate(all_words):
        if i == target_idx:
            continue  # Пропускаем само целевое слово

        if clean_word(w['word']) == answer_clean:
            word_start_ms = int(w['start'] * 1000)
            word_end_ms = int(w['end'] * 1000)

            # Если слово попадает в диапазон аудио вопроса — это spoiler
            if word_start_ms >= q_start_ms and word_end_ms <= q_end_ms:
                return True

    return False

@app.route('/start', methods=['POST'])
def start_gen():
    """Добавляет задачу генерации в очередь."""
    with job_status_lock:
        if job_status["is_busy"]:
            return jsonify({"error": "Busy"}), 400

    d = request.json
    if not d or 'game_id' not in d:
        return jsonify({"error": "game_id required"}), 400

    # Добавляем в очередь вместо создания нового потока
    task_queue.put({
        'game_id': d['game_id'],
        'token': d.get('token'),
        'urls': d.get('urls', [])
    })

    return jsonify({"msg": "Queued"})

@app.route('/status', methods=['GET'])
def get_status(): return jsonify(job_status)

# Запускаем worker-поток при импорте модуля (daemon=True — завершится вместе с главным)
_worker_thread = threading.Thread(target=task_worker, daemon=True)
_worker_thread.start()
logger.info("✅ Task worker started")


if __name__ == '__main__':
    import logging
    # Отключаем HTTP логи Werkzeug (GET /status и т.д.)
    log_werkzeug = logging.getLogger('werkzeug')
    log_werkzeug.setLevel(logging.ERROR)

    app.run(host='0.0.0.0', port=5001)
