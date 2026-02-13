import os
import json
import uuid
import re
import threading
import traceback
import shutil
import gc
import random
import warnings
import queue
import logging
import ctypes
import psutil
from collections import Counter
from flask import Flask, request, jsonify
import torch

# --- НАСТРОЙКА ЛОГИРОВАНИЯ ---
logging.basicConfig(
    level=logging.INFO,
    format='[GEN] %(message)s'
)
logger = logging.getLogger(__name__)

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
import ollama
from pydub import AudioSegment
from yandex_music import Client as YandexClient
from yandex_music.exceptions import NotFoundError

app = Flask(__name__)

# --- СЛОВАРИ И ФИЛЬТРЫ ---

JUNK_PHRASES = [
    # Технические надписи
    "субтитры", "подогнал", "сделал", "создавал", "dimatorzok",
    "динамичная", "редактор", "корректор", "подпишись",
    "симон", "продолжение", "спасибо", "просмотр", "лайк", "translated", "by",
    # Галлюцинации Whisper (русские)
    "продолжение следует", "конец фильма", "спасибо за просмотр",
    "подписывайтесь", "комментарий", "ставьте лайк",
    "редактор субтитров", "добро пожаловать", "до свидания",
    "смотрите также", "не забудьте подписаться", "ставьте лайки",
    "следующее видео", "предыдущее видео", "нажмите колокольчик",
    "в следующий раз", "спасибо что смотрите", "всем привет",
    "приятного просмотра", "до новых встреч", "оставайтесь с нами",
    # Английские артефакты и галлюцинации
    "subtitles", "lyrics", "music", "applause", "[music]", "[applause]",
    "thank you", "subscribe", "like and subscribe",
    "you", "the", "and", "is", "it", "for", "this", "that",
    "thanks for watching", "see you next time", "bye bye",
    # Повторяющиеся артефакты
    "ааа", "ооо", "ммм", "эээ", "ууу", "ля ля ля", "на на на",
    "рифма", "куплеты", "припев", "песня на русском", 
    "русском языке", "текст песни", "поэзия"
]

def is_hallucination(segment_text):
    """Детект галлюцинаций Whisper."""
    if not segment_text:
        return True
    words = segment_text.split()
    # Повторение одного слова несколько раз
    if len(words) > 3 and len(set(words)) == 1:
        return True
    # Слишком длинный сегмент (галлюцинация)
    if len(segment_text) > 500:
        return True
    # Слишком много повторов (>90% одинаковых слов)
    # Повышен порог с 70% до 90% для песен с повторяющимися фразами
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
LLM_MODEL = "gpt-oss:20b"
WHISPER_SIZE = "large-v3"
COMPUTE_TYPE = "int8"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# DSP настройки (ревью: ослаблены для лучшего качества)
DSP_HIGHPASS_FREQ = 100          # Было 200Hz — слишком агрессивно для мужского вокала
DSP_COMPRESSOR_RATIO = 2.5       # Было 4.0 — слишком жёстко, поднимает шумы
DSP_COMPRESSOR_THRESHOLD = -20.0

# Тайминги викторины
MIN_QUESTION_DURATION_MS = 20000  # Минимальная длительность вопроса
TARGET_QUESTION_DURATION_MS = 28000
MIN_AUDIO_POSITION = 12.0         # Минимальная позиция слова в секундах (spoiler protection)

# Scoring для алгоритмического выбора (score_candidates)
SCORE_EOL_BONUS = 80              # Бонус за конец строки (рифма) — ГЛАВНЫЙ ПРИОРИТЕТ
SCORE_PUNCTUATION_BONUS = 20      # Бонус за пунктуацию
SCORE_LONG_WORD_BONUS = 10        # Бонус за длинное слово (>=6 букв)
SCORE_MEDIUM_WORD_BONUS = 5       # Бонус за среднее слово (>=5 букв)
SCORE_UNIQUE_WORD_BONUS = 25      # Бонус за уникальное слово
SCORE_RARE_WORD_BONUS = 15        # Бонус за редкое слово (2 вхождения)
SCORE_FREQUENT_PENALTY = -50      # Штраф за частое слово (>=4 вхождений)
SCORE_VERB_ENDING_PENALTY = -10   # Штраф за глагольные окончания

# LLM настройки
LLM_NUM_PREDICT = 5000            # Thinking mode съедает ~1500 токенов на анализ
LLM_TEMPERATURE = 0.6

ollama_client = ollama.Client(host=OLLAMA_HOST)

if not os.path.exists(MEDIA_ROOT):
    os.makedirs(MEDIA_ROOT)

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


def log(msg):
    """Thread-safe логирование."""
    with job_status_lock:
        job_status["logs"].append(msg)
    logger.info(msg)

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

def isolate_vocals(audio_path, use_cache=True):
    """
    Изолирует вокал с помощью MDX-Net (audio-separator).
    ОПТИМИЗИРОВАНО ДЛЯ CPU: Генерирует только стем вокала.
    """
    output_dir = os.path.dirname(audio_path)
    filename = os.path.basename(audio_path)
    file_id = os.path.splitext(filename)[0]
    
    # Ожидаемый путь к файлу вокала
    expected_vocals_path = os.path.join(output_dir, f"{file_id}_vocals.wav")

    if use_cache and os.path.exists(expected_vocals_path):
        log(f"✅ Найден кэшированный вокал (MDX): {os.path.basename(expected_vocals_path)}")
        return expected_vocals_path

    log(f"🎸 Запуск MDX-Net (CPU) для {filename}...")

    try:
        from audio_separator.separator import Separator
        
        # Инициализация
        # model_file_dir - папка для кэширования моделей (монтируется в docker-compose)
        model_cache_dir = os.environ.get("MDX_MODEL_CACHE", "/app/mdx_cache")

        separator = Separator(
            log_level=logging.ERROR,
            output_dir=output_dir,
            output_format="wav",
            model_file_dir=model_cache_dir,  # Кэш моделей
            # ВАЖНО ДЛЯ CPU: Генерируем ТОЛЬКО вокал, чтобы не тратить ресурсы на инструментал
            output_single_stem="Vocals",
            # Roformer использует MDXC архитектуру — batch_size для 8845HS + 29GB RAM
            mdxc_params={"batch_size": 4}
        )

        # Mel-Band Roformer — SOTA архитектура 2025-2026
        # Даёт естественный тембр голоса (лучше для Whisper, чем MDX)
        # Альтернатива: model_bs_roformer_ep_317_sdr_12.9755.ckpt (агрессивнее, но суше)
        separator.load_model(
            model_filename='model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt'
        )

        # Запуск разделения
        output_files = separator.separate(audio_path)
        
        # Переименование результата в стандартное имя
        generated_file = None
        if output_files and len(output_files) > 0:
            generated_file = os.path.join(output_dir, output_files[0])
            
        if generated_file and os.path.exists(generated_file):
            # Удаляем старый файл если был
            if os.path.exists(expected_vocals_path):
                os.remove(expected_vocals_path)
            
            os.rename(generated_file, expected_vocals_path)
            log(f"✅ Вокал MDX извлечен: {os.path.basename(expected_vocals_path)}")
            return expected_vocals_path
        else:
            log("⚠️ Файл вокала не был создан")
            return audio_path

    except Exception as e:
        log(f"⚠️ Ошибка MDX-Net: {e}")
        # Если не вышло, пробуем вернуть оригинал, чтобы процесс не упал
        return audio_path

def generation_task(game_id, token, urls):
    with job_status_lock:
        job_status["is_busy"] = True
        job_status["progress"] = 0
        job_status["logs"] = []
        job_status["status"] = "running"

    # Статистика генерации
    generation_stats = {
        'total_tracks': 0,
        'tracks_processed': 0,
        'llm_success': 0,
        'llm_failed_validation': 0,
        'algo_fallback': 0,
        'skipped_short': 0,
        'skipped_no_context': 0,  # Пропущены из-за короткого контекста
        'questions_created': 0
    }

    game_media_folder = os.path.join(MEDIA_ROOT, f"{game_id}-media")
    game_temp_folder = os.path.join(MEDIA_ROOT, f"{game_id}-temp_downloads")
    game_json_file = os.path.join(MEDIA_ROOT, f"{game_id}-questions.json")
    
    if not os.path.exists(game_media_folder): os.makedirs(game_media_folder)
    if not os.path.exists(game_temp_folder): os.makedirs(game_temp_folder)
    
    try:
        client = None
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
                    
                    if not os.path.exists(fpath): 
                        client.tracks([tid])[0].download(fpath)
                    downloaded.append(fpath)
                    
                    # --- ПОЛУЧАЕМ ИНФОРМАЦИЮ О ТРЕКЕ ---
                    try:
                        track_info = client.tracks([tid])[0]
                        artist = track_info.artists[0].name if track_info.artists else "Неизвестен"
                        title = track_info.title
                        
                        # 1. Пробуем достать официальный текст
                        lyrics = ""
                        # Способ 1: Новый API (get_lyrics + fetch_lyrics) — 2025-2026
                        try:
                            lyrics_obj = track_info.get_lyrics('LRC')
                            if lyrics_obj:
                                raw_lyrics = lyrics_obj.fetch_lyrics()
                                lyrics = clean_lrc_lyrics(raw_lyrics)
                                log(f"📜 Найден текст (LRC) для: {title}")
                        except NotFoundError:
                            pass  # Текст не найден в новом API
                        except Exception:
                            pass  # Другая ошибка — пробуем старый способ

                        # Способ 2: Fallback на старый API (supplement)
                        if not lyrics:
                            try:
                                supp = track_info.get_supplement()
                                if supp and supp.lyrics and supp.lyrics.full_lyrics:
                                    lyrics = supp.lyrics.full_lyrics
                                    log(f"📜 Найден текст (supplement) для: {title}")
                            except Exception:
                                pass  # Текст не найден — не критично

                        # Сохраняем всё в словарь
                        files_metadata[fpath] = {
                            "meta": f"{artist} - {title}",
                            "lyrics": lyrics
                        }
                        
                        log(f"📥 Скачан: {tid} ({artist} - {title})")
                    except Exception as meta_e:
                        log(f"📥 Скачан: {tid} (без метаданных)")
                        files_metadata[fpath] = {"meta": "", "lyrics": ""}
                        
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

            # 1. MDX-Net
            recognition_source = isolate_vocals(fpath)

            # 2. DSP (Compressor)
            recognition_source = preprocess_for_whisper(recognition_source)

            try:
                # Достаем инфу
                file_data = files_metadata.get(fpath, {"meta": "", "lyrics": ""})
                
                # 3. WhisperX с "Чит-кодом" (Lyrics Prompt)
                words = process_audio_with_whisperx(
                    recognition_source, 
                    device=device, 
                    song_meta=file_data["meta"],
                    official_lyrics=file_data["lyrics"] # <--- Передаем текст!
                )
                
                if len(words) < 15:
                    log("🚫 Мало слов. Пропуск.")
                    generation_stats['skipped_short'] += 1
                    continue
                
                target_idx = -1
                used_method = "none"
                blacklist = []
                
                # 4. LLM / Algo
                for attempt in range(3):
                    llm_data = get_quiz_data_llm(words, forbidden_words=blacklist)
                    validated_data, validation_error = validate_llm_response(llm_data, words)
                    
                    if validation_error:
                        log(f"⚠️ Валидация LLM: {validation_error}")
                        generation_stats['llm_failed_validation'] += 1
                        if llm_data and 'hidden_answer' in llm_data:
                            blacklist.append(clean_word(llm_data['hidden_answer']))
                        continue

                    if validated_data:
                        ans_clean = clean_word(validated_data['hidden_answer'])
                        ctx_clean = clean_word(validated_data.get('context_snippet', ''))
                        found_idx = find_safest_occurrence_index(words, ans_clean, ctx_clean)

                        if found_idx != -1:
                            target_idx = found_idx
                            used_method = f"llm_try_{attempt+1}"
                            break
                        else:
                            log(f"⚠️ Спойлер защита: '{ans_clean}' слишком рано.")
                            blacklist.append(ans_clean)
                
                if target_idx == -1:
                    log("⚠️ Fallback: Smart Algorithm.")
                    target_idx = get_algorithmic_choice(words)
                    used_method = "algo"
                    generation_stats['algo_fallback'] += 1
                else:
                    generation_stats['llm_success'] += 1
                
                if target_idx == -1 or target_idx is None:
                    log("❌ Не удалось выбрать слово.")
                    continue

                # Пробуем найти слово с достаточным контекстом (до 5 попыток)
                tried_indices = set()
                final_target_idx = None

                for timing_attempt in range(5):
                    if target_idx in tried_indices:
                        # После 2 попыток расширяем диапазон поиска
                        extended = timing_attempt >= 2
                        ranked = score_candidates(words, extended_range=extended)
                        found_new = False
                        for candidate_idx in ranked:
                            if candidate_idx not in tried_indices:
                                target_idx = candidate_idx
                                found_new = True
                                break
                        if not found_new:
                            break  # Все варианты исчерпаны

                    tried_indices.add(target_idx)
                    q_times, a_times = calculate_timings(words, target_idx)

                    # Проверка 1: достаточная длительность контекста
                    if (q_times[1] - q_times[0]) < 20000:
                        log(f"⚠️ Контекст короткий для '{words[target_idx]['word']}', пробуем другое слово... (попытка {timing_attempt + 1}/5)")
                        continue

                    # Проверка 2: слово-ответ не должно звучать в аудио вопроса (spoiler)
                    if check_spoiler_in_question(words, target_idx, q_times[0], q_times[1]):
                        log(f"⚠️ Spoiler: '{words[target_idx]['word']}' звучит в аудио вопроса, пробуем другое... (попытка {timing_attempt + 1}/5)")
                        continue

                    # Обе проверки пройдены
                    final_target_idx = target_idx
                    break

                if final_target_idx is None:
                    log(f"❌ Не найдено слово с достаточным контекстом. Пропуск.")
                    generation_stats['skipped_no_context'] += 1
                    continue

                target_idx = final_target_idx
                answer_word_obj = words[target_idx]
                answer_text = re.sub(r'[^\w]', '', answer_word_obj['word'])

                audio = AudioSegment.from_mp3(fpath)
                q_seg = audio[q_times[0]:q_times[1]].fade_out(150).fade_in(1500)
                a_seg = audio[a_times[0]:a_times[1]].fade_in(100) 
                
                qid = str(uuid.uuid4())[:8]
                q_seg.export(os.path.join(game_media_folder, f"{qid}-1.mp3"), format="mp3")
                a_seg.export(os.path.join(game_media_folder, f"{qid}-2.mp3"), format="mp3")
                
                context_str = build_context_string(words, target_idx)
                
                new_qs.append({
                    "id": qid,
                    "type": "text",
                    "question": context_str,
                    "answer": answer_text,
                    "options": []
                })

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
        log(f"   LLM успех: {generation_stats['llm_success']}")
        log(f"   Algo fallback: {generation_stats['algo_fallback']}")
        if generation_stats['skipped_short'] > 0:
            log(f"   Пропущено (мало слов): {generation_stats['skipped_short']}")
        if generation_stats['skipped_no_context'] > 0:
            log(f"   Пропущено (короткий контекст): {generation_stats['skipped_no_context']}")

    except Exception as e:
        log(f"FATAL: {e}")
        logger.exception("Generation task failed")
    finally:
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
            
            # Параметры ASR
            "log_prob_threshold": -1.0,   
            "no_speech_threshold": 0.8,   

            "initial_prompt": prompt, # <--- Самое важное здесь
            "hallucination_silence_threshold": 2.0,
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
            threads=8  # Увеличено с 4 (половина потоков CPU)
        )

        audio = whisperx.load_audio(audio_path)

        result = model.transcribe(
            audio,
            batch_size=16,  # 8845HS имеет достаточно RAM
            language="ru",
            task="transcribe"
        )
        
        raw_segments = result["segments"]

        del model
        gc.collect()
        if device == "cuda": torch.cuda.empty_cache()

        log("Aligning audio...")
        try:
            model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
            aligned_result = whisperx.align(raw_segments, model_a, metadata, audio, device, return_char_alignments=False)
            del model_a
            gc.collect()
            if device == "cuda": torch.cuda.empty_cache()

            segments_to_process = aligned_result["segments"]
        except Exception as e:
            log(f"⚠️ Ошибка Alignment: {e}. Используем сырые сегменты.")
            segments_to_process = raw_segments

        # Освобождаем аудио-буфер после завершения всех операций с ним
        del audio
        gc.collect()

        all_words = []
        for segment in segments_to_process:
            seg_text = segment.get("text", "").lower()
            if any(junk in seg_text for junk in JUNK_PHRASES): continue
            if is_hallucination(seg_text): continue

            words_in_seg = segment.get("words", [])
            
            if not words_in_seg and "start" in segment and "end" in segment:
                log("⚠️ Fallback: используем сегмент целиком")
                all_words.append({
                    "word": segment["text"].strip(),
                    "start": segment["start"],
                    "end": segment["end"],
                    "is_eol": True,
                    "confidence": 0.5
                })
                continue

            for i, word in enumerate(words_in_seg):
                if "start" not in word or "end" not in word: continue

                confidence = word.get("score", 1.0)
                if confidence < 0.4: continue

                duration = word["end"] - word["start"]
                if duration < 0.05 or duration > 4.0: continue

                clean = clean_word(word["word"])
                if not re.match(r'^[а-яё]+$', clean): continue
                if len(clean) < 2: continue

                vowels = set('аеёиоуыэюя')
                vowel_count = sum(1 for c in clean if c in vowels)
                if len(clean) >= 8 and vowel_count == 0: continue

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
                    "confidence": confidence
                }
                all_words.append(w_obj)

        return all_words
    except Exception as e:
        log(f"WhisperX Error: {e}")
        traceback.print_exc()
        return []

# --- ИНТЕЛЛЕКТУАЛЬНЫЙ ВЫБОР ---

# Константы MIN_AUDIO_POSITION уже определены выше в секции "Scoring"
# Здесь только утилиты и функции

def clean_word(w):
    w = w.lower().replace('ё', 'е')
    return re.sub(r'[^\w]', '', w)

def score_candidates(all_words, extended_range=False):
    """Улучшенная алгоритмическая оценка слов с логированием."""
    if not all_words:
        return []

    scores = []
    total_words = len(all_words)
    freq_map = Counter([clean_word(w['word']) for w in all_words])

    # При extended_range расширяем диапазон поиска
    min_progress = 0.20 if extended_range else 0.30
    max_progress = 0.85 if extended_range else 0.70
    min_audio_pos = MIN_AUDIO_POSITION * 0.7 if extended_range else MIN_AUDIO_POSITION

    for i, w_obj in enumerate(all_words):
        word = w_obj['word']
        clean = clean_word(word)

        # Базовые фильтры
        if len(clean) < 4:
            continue
        if clean in STOP_WORDS:
            continue

        # Позиционный фильтр
        progress = i / total_words
        if progress < min_progress or progress > max_progress:
            continue

        # Временной фильтр
        if w_obj['start'] < min_audio_pos:
            continue

        score = 0

        # Бонус за конец строки (рифма)
        if w_obj.get('is_eol', False):
            score += SCORE_EOL_BONUS

        # Бонус за пунктуацию (конец фразы)
        if any(p in word for p in [',', '.', '!', '?']):
            score += SCORE_PUNCTUATION_BONUS

        # Бонус за длину слова
        if len(clean) >= 6:
            score += SCORE_LONG_WORD_BONUS
        elif len(clean) >= 5:
            score += SCORE_MEDIUM_WORD_BONUS

        # Частотный анализ (штраф за припев)
        count = freq_map[clean]
        if count == 1:
            score += SCORE_UNIQUE_WORD_BONUS
        elif count == 2:
            score += SCORE_RARE_WORD_BONUS
        elif count >= 4:
            score += SCORE_FREQUENT_PENALTY

        # Штраф за типичные глагольные окончания (менее интересны)
        if clean.endswith(('ать', 'ить', 'ять', 'еть', 'ение', 'ание')):
            score += SCORE_VERB_ENDING_PENALTY

        scores.append((score, i, clean))

    # Сортировка по убыванию очков
    scores.sort(key=lambda x: x[0], reverse=True)

    # Логирование топ-5 кандидатов для отладки
    if scores:
        top5 = [(s[2], s[0]) for s in scores[:5]]
        log(f"📊 Топ-5 кандидатов (алгоритм): {top5}")

    return [x[1] for x in scores]

def get_algorithmic_choice(all_words):
    ranked_indices = score_candidates(all_words)
    if not ranked_indices: return None
    top_n = min(len(ranked_indices), 3)
    chosen_idx = random.choice(ranked_indices[:top_n])
    return chosen_idx

def correct_transcription_with_llm(raw_text):
    if not raw_text or len(raw_text) < 20: return raw_text

    # Ограничиваем вход, чтобы не переполнять контекст
    input_slice = raw_text[:3000]

    prompt = f"""Ты — строгий редактор. Исправь явные фонетические ошибки в тексте песни.
ПРАВИЛА:
1. Сохраняй структуру строк.
2. НЕ пиши никаких вступлений, комментариев или мыслей.
3. Если текст повторяется (припев) — это нормально, оставь как есть.
4. НЕ переходи на английский язык.
5. Верни ТОЛЬКО исправленный русский текст.

Текст:
{input_slice}

Исправленный текст:"""

    try:
        response = ollama_client.chat(
            model=LLM_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            options={
                'temperature': 0.1,       # Минимальная креативность
                'num_predict': 1024,      # ЖЕСТКИЙ ЛИМИТ: макс 1024 токена на выход (хватит для песни)
                'repetition_penalty': 1.15, # ШТРАФ: давим зацикливания
                'top_p': 0.9,             # Отсекаем маловероятный бред
                'stop': ["User:", "Assistant:", "\n\n\n"] # Стоп-слова
            }
        )
        corrected = response['message']['content'].strip()

        # --- ЗАЩИТА ОТ СБОЕВ ---
        
        # 1. Проверка на английский бред (как в вашем логе)
        if "The conversation is" in corrected or "assistant is stuck" in corrected:
            log("⚠️ LLM Loop detected. Возвращаем оригинал.")
            return raw_text

        # 2. Проверка длины (если модель начала писать "войну и мир")
        if len(corrected) > len(raw_text) * 1.5:
            log("⚠️ LLM сгенерировала слишком много текста. Сброс.")
            return raw_text

        # 3. Проверка на пустоту
        if len(corrected) < 10:
            return raw_text

        return corrected

    except Exception as e:
        log(f"⚠️ Ошибка LLM коррекции: {e}")
        return raw_text

def parse_simple_response(text):
    """
    Парсит простой текстовый ответ формата:
    СЛОВО: <слово>
    КОНТЕКСТ: <контекст>
    """
    data = {}
    
    # Ищем слово (поддерживаем разные варианты написания)
    # re.IGNORECASE позволяет ловить "Слово:", "СЛОВО:", "Word:"
    word_match = re.search(r'(?:СЛОВО|ОТВЕТ|WORD):\s*([^\n]+)', text, re.IGNORECASE)
    if word_match:
        # Убираем кавычки и лишние пробелы
        data['hidden_answer'] = word_match.group(1).strip().strip('"').strip("'")
    
    # Ищем контекст
    ctx_match = re.search(r'(?:КОНТЕКСТ|CONTEXT):\s*([^\n]+)', text, re.IGNORECASE)
    if ctx_match:
        data['context_snippet'] = ctx_match.group(1).strip().strip('"').strip("'")
    else:
        # Если контекста нет - не страшно, валидатор найдет слово и так
        data['context_snippet'] = ""
        
    return data if 'hidden_answer' in data else None

def get_quiz_data_llm(all_words, forbidden_words=None):
    """
    Исправленная версия:
    1. Убрано жесткое требование к длине контекста (теперь находит слова даже в начале фразы).
    2. Добавлен агрессивный поиск слова.
    """
    # --- 1. ОЧИСТКА ТЕКСТА ОТ МУСОРА WHISPER ---
    clean_words_list = []
    # Слова-паразиты, которые Whisper любит вставлять в конец
    garbage_triggers = {"рифма", "куплеты", "припев", "песня", "русском", "языке", "текст", "поэзия"}
    
    for w in all_words:
        # Проверяем чистое слово
        check_w = re.sub(r'[^\w]', '', w['word'].lower())
        if check_w in garbage_triggers:
            continue
        clean_words_list.append(w)

    if len(clean_words_list) < 5:
        clean_words_list = all_words

    # --- 2. ПОДГОТОВКА СРЕЗА ДЛЯ LLM ---
    # Берём 30-80% песни (второй куплет и припев, избегаем начало и конец)
    mid_start = int(len(clean_words_list) * 0.30)
    mid_end = int(len(clean_words_list) * 0.80)
    slice_words = clean_words_list[mid_start:mid_end]

    if not slice_words: slice_words = clean_words_list

    # Формируем текст с переносами строк после рифм (is_eol)
    # Очищаем слова от лишней пунктуации для LLM
    text_parts = []
    for w in slice_words:
        # Убираем пунктуацию из слова, оставляя только буквы
        clean_w = re.sub(r'[^\wа-яёА-ЯЁ\-]', '', w['word'])
        if clean_w:
            text_parts.append(clean_w)
        if w.get('is_eol', False):
            text_parts.append('\n')  # Перенос после конца строки
    text_slice = " ".join(text_parts).replace(' \n ', '\n').replace(' \n', '\n')

    # Ограничиваем длину символов, чтобы не перегружать память
    if len(text_slice) > 1200:
        text_slice = text_slice[:1200].rsplit('\n', 1)[0]  # Обрезаем по строке
    log(f"📄 Отправляем в LLM: {text_slice[:200]}...")

    # --- СТАТИСТИКА РЕДКИХ СЛОВ ДЛЯ LLM ---
    # Подсчитываем частоту слов в slice_words
    freq_map = Counter([clean_word(w['word']) for w in slice_words])

    # Собираем уникальные слова (1-2 вхождения), длиннее 4 букв, в конце строки
    rare_words = []
    for w in slice_words:
        w_clean = clean_word(w['word'])
        if len(w_clean) >= 5 and freq_map[w_clean] <= 2 and w.get('is_eol', False):
            if w_clean not in rare_words:
                rare_words.append(w_clean)

    # Ограничиваем до 15 слов
    rare_words = rare_words[:15]
    rare_words_str = ", ".join(rare_words) if rare_words else "нет данных"

    blacklist_instruction = ""
    if forbidden_words:
        blacklist_instruction = f"Исключи слова: {', '.join(forbidden_words)}. "

    # --- 3. УЛУЧШЕННЫЙ ПРОМПТ С ПРИМЕРАМИ И СТАТИСТИКОЙ ---
    prompt = f"""Текст песни (каждая строка — отдельная фраза):
{text_slice}

РЕДКИЕ СЛОВА (встречаются 1-2 раза, идеальны для викторины):
{rare_words_str}

Выбери слово-РИФМУ для викторины.

РИФМА = ПОСЛЕДНЕЕ слово в строке. Примеры:
"Бабки текут рекой" → рифма: РЕКОЙ
"Будешь местный кент" → рифма: КЕНТ
"По ядерной долине" → рифма: ДОЛИНЕ

Требования:
1. ПРИОРИТЕТ: выбирай из списка РЕДКИХ СЛОВ выше
2. Выбирай ПОСЛЕДНЕЕ слово строки (перед переносом)
3. Существительное длиннее 4 букв
4. {blacklist_instruction}Из второй половины текста

Напиши ОДНО слово:"""

    try:
        # --- 4. ЗАПРОС ---
        response = ollama_client.chat(
            model=LLM_MODEL,
            messages=[{'role': 'user', 'content': prompt}],
            options={
                'temperature': 0.6,
                'num_predict': LLM_NUM_PREDICT,  # thinking mode съедает ~1500 токенов
                'top_p': 0.9
            }
        )

        raw_answer = response['message']['content'].strip()

        # При пустом ответе — выводим полный response для диагностики
        if not raw_answer:
            log(f"⚠️ LLM вернула пустой content. Full response: {response}")
            return None

        # Очистка ответа (убираем точки, кавычки, берем последнее слово)
        # ВАЖНО: явно указываем кириллицу, т.к. \w может не включать её на Windows
        clean_resp = re.sub(r'[^а-яёА-ЯЁa-zA-Z0-9\-]', ' ', raw_answer)
        resp_words = clean_resp.split()

        if not resp_words:
            log(f"⚠️ LLM вернула пустоту после очистки. Raw: '{raw_answer[:50]}'")
            return None
            
        target_word = resp_words[-1] 
        log(f"🤖 LLM выбрала: {target_word}")

        # --- 5. ПОИСК КОНТЕКСТА (ИСПРАВЛЕННЫЙ) ---
        found_idx = -1
        target_clean = clean_word(target_word) # Используем вашу clean_word (с ё->е)
        
        # Ищем слово в списке
        for i, w in enumerate(slice_words):
            current_clean = clean_word(w['word'])
            
            # Сравнение
            if current_clean == target_clean:
                found_idx = i
                # Если у нас есть хотя бы 1 слово контекста - уже хорошо.
                # Если это самое первое слово (i=0), ищем дальше, может оно повторяется.
                if i >= 1: 
                    break 
        
        # Если точное совпадение не найдено, пробуем "мягкий" поиск (вхождение)
        if found_idx == -1:
            for i, w in enumerate(slice_words):
                current_clean = clean_word(w['word'])
                if target_clean in current_clean and len(current_clean) < len(target_clean) + 3:
                    log(f"⚠️ Fuzzy match: '{target_word}' -> '{w['word']}'")
                    found_idx = i
                    break

        if found_idx != -1:
            # Формируем контекст
            # Берем до 4 слов перед ответом, но не выходим за границы списка
            ctx_start = max(0, found_idx - 4)
            snippet_words = [w['word'] for w in slice_words[ctx_start:found_idx]]
            snippet = " ".join(snippet_words)
            
            # Если контекст пустой (слово первое в списке), берем само слово как заглушку
            # (Валидатор это не пропустит, но программа не упадет)
            if not snippet:
                snippet = "..." 

            return {
                "hidden_answer": target_word,
                "context_snippet": snippet
            }
        else:
            # Если все равно не нашли - выводим список слов для отладки
            debug_list = [clean_word(w['word']) for w in slice_words[:10]]
            log(f"⚠️ Слово '{target_word}' ({target_clean}) не найдено. Первые 10 слов среза: {debug_list}")
            return None

    except Exception as e:
        log(f"⚠️ LLM Error: {e}")
        traceback.print_exc()
        
    return None

def validate_llm_response(data, all_words):
    """Валидация ответа LLM перед использованием."""
    if not data:
        return None, "Пустой ответ"

    if "hidden_answer" not in data:
        return None, "Нет поля hidden_answer"

    # Очищаем ответ от случайных точек/запятых, которые могли прилипнуть в Regex
    answer = data.get("hidden_answer", "").strip(" .,!?;:\"'")

    # Проверка на пробелы (нам нужны только одиночные слова)
    if ' ' in answer:
        return None, f"Ответ '{answer}' содержит несколько слов"

    # Очищаем для сравнения
    clean = clean_word(answer)

    # Проверка длины
    if len(clean) < 4:
        return None, f"Слово '{answer}' слишком короткое ({len(clean)} букв)"

    # Собираем список слов из Whisper, тоже нормализуя (ё->е)
    text_words = set([clean_word(w['word']) for w in all_words])
    
    # ГЛАВНАЯ ПРОВЕРКА: Есть ли слово в тексте Whisper?
    if clean not in text_words:
        # Попытка найти слово, если LLM вернула его в другой форме (редко, но бывает)
        # Например, Whisper дал "стеной", а LLM "стена".
        # Но для викторины важно точное совпадение, так что лучше отклонить.
        return None, f"Слово '{answer}' (clean: {clean}) не найдено в тексте Whisper"

    # Проверка на очевидные устойчивые выражения
    context = data.get("context_snippet", "").lower()
    obvious_phrases = [
        ("стрелки на", "часах"), ("любовь в", "глазах"), ("звёзды в", "небесах"),
        ("день за", "днём"), ("раз за", "разом"), ("шаг за", "шагом"),
        ("рука в", "руке"), ("голос в", "голове"), ("дождь за", "окном"),
        ("ночь за", "ночью"), ("слёзы на", "глазах"), ("солнце в", "небе")
    ]
    for prefix, suffix in obvious_phrases:
        if prefix in context and clean == clean_word(suffix):
            return None, f"Устойчивое выражение '{prefix} {suffix}' слишком очевидно"

    # Проверка на гласные (защита от мусора типа "кгб", "вдв" или галлюцинаций)
    vowels = set('аеиоуыэюяaeiouy') # ё уже заменили на е
    vowel_count = sum(1 for c in clean if c in vowels)
    if vowel_count == 0 and len(clean) > 3:
        return None, f"Слово '{answer}' не содержит гласных"

    # Проверка на нагромождение согласных (защита от сбоев кодировки или бреда)
    consonant_streak = 0
    max_streak = 0
    for c in clean:
        if c not in vowels and c.isalpha():
            consonant_streak += 1
            max_streak = max(max_streak, consonant_streak)
        else:
            consonant_streak = 0
    
    if max_streak > 5:
        return None, f"Слово '{answer}' выглядит некорректно (много согласных подряд)"

    # Если всё ок, возвращаем очищенный от пунктуации answer внутри data
    data["hidden_answer"] = answer
    return data, None

def find_safest_occurrence_index(all_words, answer_clean, context_snippet_clean=None):
    """
    Находит безопасный индекс слова (после MIN_AUDIO_POSITION секунд),
    чтобы избежать спойлеров в начале песни.
    """
    candidates = []
    for i, w in enumerate(all_words):
        if clean_word(w['word']) == answer_clean:
            candidates.append(i)

    if not candidates:
        return -1

    # Сначала пробуем найти по контексту (если LLM дала подсказку)
    if context_snippet_clean:
        snippet_words = context_snippet_clean.split()
        last_context_word = snippet_words[-1] if snippet_words else ""
        for idx in candidates:
            if idx > 0 and all_words[idx]['start'] > MIN_AUDIO_POSITION:
                prev_w = clean_word(all_words[idx-1]['word'])
                if prev_w in last_context_word or last_context_word in prev_w:
                    return idx

    # Ищем первое вхождение после MIN_AUDIO_POSITION
    for idx in candidates:
        if all_words[idx]['start'] > MIN_AUDIO_POSITION:
            # Если слово частое (>3 раз), предупреждаем но разрешаем
            if len(candidates) > 3:
                log(f"🛡️ Spoiler Protection: Слово '{answer_clean}' частое ({len(candidates)}x), выбрано вхождение на {all_words[idx]['start']:.1f}с")
            return idx

    # Все вхождения слишком рано
    log(f"⛔ Spoiler Protection: Все вхождения '{answer_clean}' раньше {MIN_AUDIO_POSITION}с")
    return -1

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

# --- СБОРКА ---

def calculate_timings(all_words, target_idx):
    """Вычисляет тайминги для вопроса и ответа."""
    if not all_words or target_idx < 0 or target_idx >= len(all_words):
        # Защита от пустого списка или некорректного индекса
        return (0, 0), (0, 0)

    target_word = all_words[target_idx]
    cut_ms = int(target_word['start'] * 1000)
    
    # --- НАСТРОЙКИ ДЛИТЕЛЬНОСТИ ---
    TARGET_DURATION = 28000  # Целимся в 28 секунд
    MIN_DURATION = 25000     # Жесткий минимум 25 секунд
    # ------------------------------

    raw_start_ms = cut_ms - TARGET_DURATION
    if raw_start_ms < 0: 
        raw_start_ms = 0
    
    q_start_ms = raw_start_ms
    
    # Логика "умного" старта: ищем начало слова, которое близко к raw_start_ms,
    # но при этом не делает фрагмент короче MIN_DURATION.
    best_diff = float('inf')

    for w in all_words:
        w_ms = int(w['start'] * 1000)

        # Вычисляем, какой длины будет фрагмент, если начать с этого слова
        duration_candidate = cut_ms - w_ms

        # Если фрагмент становится короче минимума - это слово уже слишком близко к ответу,
        # прерываем поиск, чтобы не взять его.
        if duration_candidate < MIN_DURATION:
            break

        # Ищем слово, старт которого ближе всего к нашему идеальному raw_start_ms
        diff = abs(w_ms - raw_start_ms)

        # Если находим кандидата в пределах 3-х секунд от идеального старта
        if diff < 3000 and diff < best_diff:
            best_diff = diff
            # Берем старт слова минус небольшой отступ (100мс), чтобы не глотать начало
            q_start_ms = max(0, w_ms - 100)

    q_end_ms = max(0, cut_ms - 150) # Конец чуть перед словом
    
    # Финальная проверка: если вдруг фрагмент получился подозрительно коротким
    # (например, из-за сбоя логики или очень раннего слова), форсируем длину.
    if (q_end_ms - q_start_ms) < MIN_DURATION:
        # Пытаемся отступить назад на 26 секунд от ответа
        q_start_ms = max(0, cut_ms - 26000)
        
    # Настройки для ОТВЕТА: 5 сек до ответа + 10-15 сек после (чтобы был слышен ответ)
    a_start_ms = max(0, cut_ms - 5000)  # 5 секунд до слова-ответа
    a_duration_ms = 15000  # 15 секунд всего (5 сек до + 10 сек после ответа)
    a_end_ms = min(int(all_words[-1]['end']*1000) + 1000, a_start_ms + a_duration_ms)
    
    return (q_start_ms, q_end_ms), (a_start_ms, a_end_ms)

def build_context_string(all_words, target_idx):
    start_idx = max(0, target_idx - 12)
    current_ctx_words = []
    for i in range(target_idx - 1, start_idx - 1, -1):
        curr_w = all_words[i]
        if i > 0:
            prev_w = all_words[i-1]
            if (curr_w['start'] - prev_w['end']) > 1.2:
                current_ctx_words.insert(0, curr_w['word'])
                break
        current_ctx_words.insert(0, curr_w['word'])
    
    if len(current_ctx_words) < 2:
        s = max(0, target_idx - 8)
        current_ctx_words = [w['word'] for w in all_words[s:target_idx]]

    return " ".join(current_ctx_words) + " ___"

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