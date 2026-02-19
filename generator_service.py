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
import time
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


def parse_lrc_with_timestamps(lrc_text):
    """
    Парсит LRC текст, сохраняя таймкоды строк.
    Возвращает список: [{"time_ms": 15500, "text": "Текст строки"}, ...]
    """
    if not lrc_text:
        return []

    lines = []
    for raw_line in lrc_text.split('\n'):
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        # Ищем таймкод [MM:SS.xx]
        m = re.match(r'\[(\d{2}):(\d{2})(?:\.(\d{2,3}))?\](.*)', raw_line)
        if m:
            minutes = int(m.group(1))
            seconds = int(m.group(2))
            millis_str = m.group(3) or "0"
            # Нормализуем миллисекунды (2 цифры -> *10, 3 цифры -> как есть)
            if len(millis_str) == 2:
                millis = int(millis_str) * 10
            else:
                millis = int(millis_str)
            time_ms = minutes * 60000 + seconds * 1000 + millis
            text = m.group(4).strip()
            if text:
                lines.append({"time_ms": time_ms, "text": text})
    return lines


def extract_rhyme_words_from_lyrics(lyrics_text):
    """
    Извлекает последние слова каждой строки из официального текста.
    Это гарантированно слова-рифмы.
    Возвращает список: [{"word": "рекой", "line": "Бабки текут рекой", "line_idx": 3}, ...]
    """
    if not lyrics_text:
        return []

    results = []
    lines = [l.strip() for l in lyrics_text.split('\n') if l.strip()]

    for idx, line in enumerate(lines):
        # Убираем пунктуацию в конце строки и берём последнее слово
        words_in_line = re.findall(r'[а-яёА-ЯЁa-zA-Z]+', line)
        if not words_in_line:
            continue
        last_word = words_in_line[-1].lower().replace('ё', 'е')
        if len(last_word) >= 2:
            results.append({
                "word": last_word,
                "line": line,
                "line_idx": idx
            })

    return results


def get_russian_syllable_tail(word, n=3):
    """
    Возвращает последние n символов, начиная с последней гласной.
    Используется для фонетического сравнения рифм.
    Пример: "рекой" -> "ой", "долине" -> "ине"
    """
    vowels = set('аеёиоуыэюя')
    word = word.lower().replace('ё', 'е')
    # Находим позицию последней гласной
    last_vowel_pos = -1
    for i in range(len(word) - 1, -1, -1):
        if word[i] in vowels:
            last_vowel_pos = i
            break
    if last_vowel_pos == -1:
        return word[-n:]  # Нет гласных — берём хвост
    # Берём от последней гласной до конца
    tail = word[last_vowel_pos:]
    # Если слишком короткий хвост, расширяем
    if len(tail) < 2 and last_vowel_pos > 0:
        # Ищем предпоследнюю гласную
        for i in range(last_vowel_pos - 1, -1, -1):
            if word[i] in vowels:
                tail = word[i:]
                break
    return tail


def score_lyrics_rhyme_candidates(rhyme_words, total_lines):
    """
    Алгоритмический скоринг кандидатов-рифм из официального текста.
    Не использует LLM — чисто алгоритмический подход.

    Возвращает отсортированный список: [(score, rhyme_word_entry), ...]
    """
    if not rhyme_words or total_lines == 0:
        return []

    # Подсчёт частотности
    freq_map = Counter([rw["word"] for rw in rhyme_words])

    scored = []
    for rw in rhyme_words:
        word = rw["word"]
        line_idx = rw["line_idx"]
        clean = clean_word(word)
        score = 0

        # --- Позиционный фильтр: 25-80% песни ---
        progress = line_idx / total_lines
        if progress < 0.25 or progress > 0.80:
            continue

        # --- Базовые фильтры ---
        if len(clean) < 4:
            continue
        if clean in STOP_WORDS:
            continue

        # Проверка на гласные
        vowels = set('аеиоуыэюя')
        if not any(c in vowels for c in clean):
            continue

        # --- Скоринг ---

        # Длина слова
        if len(clean) >= 7:
            score += 20
        elif len(clean) >= 6:
            score += SCORE_LONG_WORD_BONUS
        elif len(clean) >= 5:
            score += SCORE_MEDIUM_WORD_BONUS

        # Уникальность (ключевой фактор — уникальные слова интереснее)
        count = freq_map[word]
        if count == 1:
            score += SCORE_UNIQUE_WORD_BONUS + 10  # Усиленный бонус
        elif count == 2:
            score += SCORE_RARE_WORD_BONUS
        elif count >= 4:
            score += SCORE_FREQUENT_PENALTY  # Припев — скучно

        # Штраф за банальные глагольные окончания
        if clean.endswith(('ать', 'ить', 'ять', 'еть', 'ует', 'ает')):
            score += SCORE_VERB_ENDING_PENALTY

        # Бонус за существительные (эвристика по окончаниям)
        if clean.endswith(('ость', 'ство', 'ение', 'ание')):
            score += 5  # Абстрактные существительные — хорошие ответы

        # Бонус за фонетическую "рифмопригодность"
        # Проверяем, рифмуется ли с соседними строками
        tail = get_russian_syllable_tail(clean)
        rhyme_pairs = 0
        for other_rw in rhyme_words:
            if other_rw["line_idx"] != line_idx:
                other_tail = get_russian_syllable_tail(other_rw["word"])
                if tail == other_tail and len(tail) >= 2:
                    rhyme_pairs += 1
        if rhyme_pairs > 0:
            score += 15  # Рифмуется с другой строкой — игрок сможет угадать!

        scored.append((score, rw))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def _levenshtein_ratio(s1, s2):
    """Быстрое нечёткое сравнение двух строк (0.0 — разные, 1.0 — идентичны)."""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    # Оптимизация: если длины сильно различаются — не совпадение
    if abs(len1 - len2) > max(len1, len2) * 0.5:
        return 0.0
    # Простой Levenshtein через две строки матрицы
    prev = list(range(len2 + 1))
    for i in range(1, len1 + 1):
        curr = [i] + [0] * len2
        for j in range(1, len2 + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            curr[j] = min(curr[j-1] + 1, prev[j] + 1, prev[j-1] + cost)
        prev = curr
    distance = prev[len2]
    max_len = max(len1, len2)
    return 1.0 - distance / max_len


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


def select_word_from_lyrics_algorithmically(official_lyrics, all_words):
    """
    Главная функция: алгоритмически выбирает слово-рифму из официального текста.
    Затем находит его в массиве all_words (Whisper/alignment) для получения таймкодов.

    Возвращает (target_idx, answer_line) — индекс в all_words и строку текста с ответом,
    или (-1, "") если не удалось.
    """
    lines = [l.strip() for l in official_lyrics.split('\n') if l.strip()]
    total_lines = len(lines)

    if total_lines < 5:
        return -1, ""

    # 1. Извлекаем рифмы
    rhyme_words = extract_rhyme_words_from_lyrics(official_lyrics)
    if not rhyme_words:
        return -1, ""

    # 2. Скорим кандидатов
    scored = score_lyrics_rhyme_candidates(rhyme_words, total_lines)
    if not scored:
        return -1, ""

    # Логируем топ-5
    top5 = [(s[1]["word"], s[0]) for s in scored[:5]]
    log(f"📊 Топ-5 рифм (из текста): {top5}")

    # 3. Пробуем найти каждого кандидата в all_words (с таймкодами)
    for score_val, rw in scored[:10]:  # Проверяем топ-10
        target_clean = clean_word(rw["word"])

        # Ищем в all_words
        for i, w in enumerate(all_words):
            if clean_word(w['word']) == target_clean and w['start'] > MIN_AUDIO_POSITION:
                # Проверяем, что слово в допустимой позиции (25-85% массива)
                progress = i / len(all_words)
                if not (0.25 <= progress <= 0.85):
                    continue

                # Таймкод не сжат (слово длится >0.15с)
                word_dur = w['end'] - w['start']
                if word_dur < 0.15:
                    continue

                # Таймкод не раздут (>5с на слово = баг alignment)
                if word_dur > 5.0:
                    continue

                # Перекрытие с соседними словами (alignment сжат)
                if i > 0 and w['start'] < all_words[i - 1]['end'] - 0.05:
                    continue
                if i + 1 < len(all_words) and w['end'] > all_words[i + 1]['start'] + 0.05:
                    continue

                # Нет большой паузы перед словом (проигрыш >5с)
                if i > 0:
                    gap = w['start'] - all_words[i - 1]['end']
                    if gap > 5.0:
                        continue

                log(f"✅ [LYRICS_ALGO] Выбрано: '{rw['word']}' (score={score_val}, line: '{rw['line'][:40]}...')")
                return i, rw["line"]

    log("⚠️ Ни один кандидат из текста не найден в Whisper-массиве")
    return -1, ""


def select_word_from_lrc(raw_lrc, official_lyrics):
    """
    Выбирает слово-рифму используя LRC-таймкоды напрямую (без forced alignment).
    LRC даёт точные таймкоды строк от Яндекс Музыки.

    Возвращает dict: {"word", "line", "line_start_ms", "next_line_start_ms", "context"}
    или None если не удалось.
    """
    lrc_lines = parse_lrc_with_timestamps(raw_lrc)
    if len(lrc_lines) < 5:
        return None

    # Извлекаем рифмы и скорим
    rhyme_words = extract_rhyme_words_from_lyrics(official_lyrics)
    if not rhyme_words:
        return None

    scored = score_lyrics_rhyme_candidates(rhyme_words, len(lrc_lines))
    if not scored:
        return None

    top5 = [(s[1]["word"], s[0]) for s in scored[:5]]
    log(f"📊 Топ-5 рифм (из текста): {top5}")

    # Для каждого кандидата ищем его строку в LRC
    for score_val, rw in scored[:10]:
        target_word = rw["word"]
        target_line = rw["line"].strip().lower()
        target_clean = clean_word(target_word)
        target_line_idx = rw["line_idx"]

        # Ищем строку в LRC-данных.
        # Стратегия: ищем LRC-строку, которая заканчивается на target_word
        # и находится в правильной позиции (25-85% трека).
        # Если слово повторяется (припев), берём ту что ближе к середине.
        candidates_lrc = []
        for li, ll in enumerate(lrc_lines):
            lrc_text = ll["text"].strip().lower()
            lrc_words = re.findall(r'[а-яёА-ЯЁa-zA-Z]+', lrc_text)
            if not lrc_words:
                continue

            lrc_last_word = clean_word(lrc_words[-1])

            # Последнее слово LRC-строки = наш target
            if lrc_last_word == target_clean:
                candidates_lrc.append(li)
            # Fallback: подстрочное сравнение
            elif target_line in lrc_text or lrc_text in target_line:
                candidates_lrc.append(li)

        if not candidates_lrc:
            continue

        # Из кандидатов выбираем тот, что в правильном диапазоне (25-85%)
        # и ближе к пропорциональной позиции target_line_idx
        total_dur_ms_approx = lrc_lines[-1]["time_ms"] + 5000
        best_lrc_idx = None
        best_score = float('inf')
        for li in candidates_lrc:
            progress = lrc_lines[li]["time_ms"] / total_dur_ms_approx if total_dur_ms_approx > 0 else 0
            if progress < 0.20 or progress > 0.88:
                continue
            if lrc_lines[li]["time_ms"] < MIN_AUDIO_POSITION * 1000:
                continue
            # Чем ближе к пропорциональной позиции — тем лучше
            expected_progress = target_line_idx / max(1, len(lrc_lines))
            dist = abs(progress - expected_progress)
            if dist < best_score:
                best_score = dist
                best_lrc_idx = li

        if best_lrc_idx is None:
            continue

        lrc_idx = best_lrc_idx
        lrc_line = lrc_lines[lrc_idx]
        line_start_ms = lrc_line["time_ms"]

        # Таймкод следующей строки = конец текущей
        if lrc_idx + 1 < len(lrc_lines):
            next_line_start_ms = lrc_lines[lrc_idx + 1]["time_ms"]
        else:
            next_line_start_ms = line_start_ms + 5000

        # Контекст: ТОЛЬКО предыдущие строки (БЕЗ строки с ответом!)
        # Берём столько строк назад, чтобы контекст длился 20-30 сек
        MIN_CONTEXT_DURATION = 20000  # 20 сек минимум
        TARGET_CONTEXT_DURATION = 28000  # 28 сек цель

        context_parts = []
        context_start_idx = lrc_idx  # Будем двигать назад
        for ci in range(lrc_idx - 1, -1, -1):
            ctx_duration = line_start_ms - lrc_lines[ci]["time_ms"]
            if ctx_duration > TARGET_CONTEXT_DURATION:
                break
            context_start_idx = ci
            context_parts.insert(0, lrc_lines[ci]["text"].strip())

        context = " ".join(context_parts)
        context_start_ms = lrc_lines[context_start_idx]["time_ms"] if context_start_idx < lrc_idx else max(0, line_start_ms - TARGET_CONTEXT_DURATION)

        # Проверяем что контекст не слишком короткий
        context_duration = line_start_ms - context_start_ms
        if context_duration < MIN_CONTEXT_DURATION and context_start_ms > 0:
            context_start_ms = max(0, line_start_ms - MIN_CONTEXT_DURATION)

        # Проверяем что контекст не пустой
        if not context.strip():
            continue

        # Spoiler-проверка: слово-ответ не должно звучать в контексте
        if target_clean in clean_word(context):
            log(f"⚠️ [LRC_DIRECT] Spoiler: '{target_word}' найдено в контексте, пропуск")
            continue

        log(f"✅ [LRC_DIRECT] Выбрано: '{target_word}' (score={score_val}, "
            f"line_start={line_start_ms}ms, next_line={next_line_start_ms}ms, "
            f"context_start={context_start_ms}ms)")

        return {
            "word": target_word,
            "line": rw["line"],
            "line_start_ms": line_start_ms,
            "next_line_start_ms": next_line_start_ms,
            "context_start_ms": context_start_ms,
            "context": context,
            "score": score_val
        }

    log("⚠️ [LRC_DIRECT] Ни один кандидат не найден в LRC-строках")
    return None


def calculate_timings_from_lrc(lrc_result, audio_duration_ms):
    """
    Вычисляет тайминги для вопроса и ответа на основе LRC-таймкодов.

    Ключевой момент: LRC даёт только начало СТРОКИ, а ответ — ПОСЛЕДНЕЕ СЛОВО строки.
    Аудио вопроса должно включать ВСЮ строку ответа КРОМЕ последнего слова.
    Оцениваем позицию последнего слова пропорционально: (N-1)/N * line_duration.
    """
    answer_line_start_ms = lrc_result["line_start_ms"]
    next_line_ms = lrc_result["next_line_start_ms"]
    context_start_ms = lrc_result.get("context_start_ms", max(0, answer_line_start_ms - 28000))
    answer_line_text = lrc_result.get("line", "")

    # Считаем слова в строке ответа
    line_words = re.findall(r'[а-яёА-ЯЁa-zA-Z]+', answer_line_text)
    n_words = max(len(line_words), 2)  # Минимум 2 слова

    # Длительность строки ответа
    line_duration_ms = next_line_ms - answer_line_start_ms
    # Ограничиваем разумным пределом (строка не может быть >15с обычно)
    line_duration_ms = min(line_duration_ms, 15000)

    # Оценка позиции последнего слова: (N-1)/N * duration
    # Минус 300мс буфер чтобы не зацепить начало последнего слова
    estimated_answer_offset = int(line_duration_ms * (n_words - 1) / n_words) - 300
    estimated_answer_offset = max(estimated_answer_offset, int(line_duration_ms * 0.5))  # минимум 50% строки

    # q_end = начало строки + смещение до последнего слова
    q_end_ms = answer_line_start_ms + estimated_answer_offset

    # q_start = начало контекста
    q_start_ms = context_start_ms

    log(f"✂️ LRC cut: строка {n_words} слов, duration={line_duration_ms}ms, "
        f"answer_offset={estimated_answer_offset}ms, q_end={q_end_ms}ms")

    # Ответ: от 3с до q_end, до max 15с
    a_start_ms = max(0, q_end_ms - 3000)
    a_end_ms = min(audio_duration_ms, a_start_ms + 15000)

    return (q_start_ms, q_end_ms), (a_start_ms, a_end_ms)


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

def preprocess_for_alignment(audio_path):
    """
    Облегчённая DSP обработка для forced alignment.
    Forced alignment менее чувствителен к шумам, чем ASR,
    поэтому достаточно базовой обработки.
    """
    try:
        from pydub.effects import normalize

        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_frame_rate(16000).set_channels(1)

        # Мягкий high-pass (убираем только суб-бас)
        audio = audio.high_pass_filter(80)
        audio = normalize(audio)

        output_path = os.path.splitext(audio_path)[0] + '_align.wav'
        audio.export(output_path, format='wav')
        log(f"🎵 DSP (alignment mode): лёгкая обработка")
        return output_path

    except Exception as e:
        log(f"⚠️ Ошибка DSP alignment: {e}")
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
                mdxc_params={"batch_size": 8}
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

def _select_word_llm_flow(words, generation_stats):
    """
    Старый пайплайн выбора слова: LLM (3 попытки) → алгоритмический fallback.
    Возвращает (target_idx, used_method).
    """
    target_idx = -1
    used_method = "none"
    blacklist = []

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

    return target_idx, used_method


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
        'llm_success': 0,
        'llm_failed_validation': 0,
        'algo_fallback': 0,
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
                        for attempt in range(3):
                            try:
                                client.tracks([tid])[0].download(fpath)
                                break
                            except Exception as dl_e:
                                if attempt < 2:
                                    log(f"⚠️ Попытка {attempt+1}/3 не удалась для {tid}: {dl_e}. Повтор через {2 ** attempt}с...")
                                    time.sleep(2 ** attempt)
                                else:
                                    raise dl_e
                    downloaded.append(fpath)
                    
                    # --- ПОЛУЧАЕМ ИНФОРМАЦИЮ О ТРЕКЕ ---
                    try:
                        track_info = None
                        for attempt in range(3):
                            try:
                                track_info = client.tracks([tid])[0]
                                break
                            except Exception as ti_e:
                                if attempt < 2:
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
                                lyrics_obj = track_info.get_lyrics('LRC')
                                if lyrics_obj:
                                    raw_lrc = lyrics_obj.fetch_lyrics()
                                    lyrics = clean_lrc_lyrics(raw_lrc)
                                    lyrics_source = "lrc"
                                    log(f"📜 Найден текст (LRC) для: {title}")
                                break  # Успех или текста нет — не повторяем
                            except NotFoundError:
                                break  # Текст не найден — повторять бессмысленно
                            except Exception as lrc_e:
                                if attempt < 2:
                                    log(f"⚠️ LRC {tid}: попытка {attempt+1}/3 — {lrc_e}. Повтор через {2 ** attempt}с...")
                                    time.sleep(2 ** attempt)
                                else:
                                    log(f"⚠️ LRC {tid}: все 3 попытки неудачны, пробуем supplement")

                        # Способ 2: Fallback на старый API (supplement)
                        if not lyrics:
                            for attempt in range(3):
                                try:
                                    supp = track_info.get_supplement()
                                    if supp and supp.lyrics and supp.lyrics.full_lyrics:
                                        lyrics = supp.lyrics.full_lyrics
                                        lyrics_source = "supplement"
                                        log(f"📜 Найден текст (supplement) для: {title}")
                                    break
                                except Exception as sup_e:
                                    if attempt < 2:
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

                # 4. ВЫБОР СЛОВА: lyrics → алгоритмический, без lyrics → LLM
                if has_lyrics:
                    log(f"📜 Режим: WHISPER + LYRICS ALGO (текст из {file_data['lyrics_source']})")

                    # 4a. Совмещаем Whisper-таймкоды с точным текстом из lyrics
                    words = align_words_to_lyrics(words, file_data["lyrics"])

                    # 4b. Алгоритмический выбор рифмы из текста, таймкоды из Whisper
                    target_idx, answer_line = select_word_from_lyrics_algorithmically(file_data["lyrics"], words)
                    used_method = "lyrics_algo"

                    if target_idx == -1:
                        log("⚠️ Lyrics algo не нашёл кандидата, fallback на LLM...")
                        target_idx, used_method = _select_word_llm_flow(words, generation_stats)
                    else:
                        generation_stats['lyrics_algo_success'] += 1

                else:
                    log(f"🎤 Режим: WHISPER + LLM (текст не найден)")

                    # LLM / Algo выбор слова
                    target_idx, used_method = _select_word_llm_flow(words, generation_stats)

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
                    q_times, a_times = calculate_timings(words, target_idx, is_lyrics_mode=has_lyrics, answer_line=answer_line)

                    # Проверка 1: достаточная длительность контекста
                    if (q_times[1] - q_times[0]) < 20000:
                        log(f"⚠️ Контекст короткий для '{words[target_idx]['word']}', пробуем другое слово... (попытка {timing_attempt + 1}/5)")
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
                q_seg = audio[q_times[0]:q_times[1] - 100].fade_out(150).fade_in(1500)
                a_seg = audio[a_times[0]:a_times[1]].fade_in(100)
                log(f"🔍 Оригинал: {orig_duration_ms}ms, q=[{q_times[0]}-{q_times[1]}ms] ({len(q_seg)}ms), a=[{a_times[0]}-{a_times[1]}ms] ({len(a_seg)}ms)")
                
                qid = str(uuid.uuid4())[:8]
                q_seg.export(os.path.join(game_media_folder, f"{qid}-1.mp3"), format="mp3")
                a_seg.export(os.path.join(game_media_folder, f"{qid}-2.mp3"), format="mp3")
                
                context_str = build_context_string(words, target_idx, is_lyrics_mode=has_lyrics, answer_line=answer_line)
                
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
        if generation_stats['lyrics_algo_success'] > 0:
            log(f"   Lyrics Algo (Whisper + текст): {generation_stats['lyrics_algo_success']}")
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
        # Освобождаем кэшированную модель Separator
        release_separator()
        with job_status_lock:
            job_status["is_busy"] = False
            job_status["status"] = "finished"

def forced_align_lyrics(audio_path, official_lyrics, device="cpu", raw_lrc=""):
    """
    Forced alignment: вместо распознавания речи выравнивает ИЗВЕСТНЫЙ текст по аудио.
    Слова гарантированно правильные (из Яндекс Музыки), нужны только таймкоды.

    Если raw_lrc передан — используем LRC-таймкоды как точные якоря для сегментов.
    Если нет — равномерное распределение (менее точно).
    """
    try:
        log("🎯 Forced Alignment: используем официальный текст (без ASR)")

        audio = whisperx.load_audio(audio_path)
        audio_duration = len(audio) / 16000  # WhisperX загружает в 16kHz

        fake_segments = []

        # --- Вариант 1: Есть LRC с точными таймкодами от Яндекса ---
        if raw_lrc:
            lrc_lines = parse_lrc_with_timestamps(raw_lrc)
            if lrc_lines:
                log(f"🎯 Используем LRC-якоря ({len(lrc_lines)} строк с таймкодами)")
                for i, lrc_line in enumerate(lrc_lines):
                    clean_line = lrc_line["text"].strip()
                    if not clean_line:
                        continue
                    if is_junk_segment(clean_line):
                        continue

                    seg_start = lrc_line["time_ms"] / 1000.0
                    # Конец сегмента = начало следующей строки
                    if i + 1 < len(lrc_lines):
                        seg_end = lrc_lines[i + 1]["time_ms"] / 1000.0
                    else:
                        seg_end = min(seg_start + 10.0, audio_duration)

                    fake_segments.append({
                        "text": clean_line,
                        "start": seg_start,
                        "end": seg_end
                    })

        # --- Вариант 2: Нет LRC — равномерное распределение (fallback) ---
        if not fake_segments:
            lines = [l.strip() for l in official_lyrics.split('\n') if l.strip()]
            if not lines:
                return []

            log(f"🎯 LRC отсутствует, равномерное распределение ({len(lines)} строк)")
            time_per_line = audio_duration / len(lines)
            for i, line in enumerate(lines):
                clean_line = line.strip()
                if not clean_line:
                    continue
                if is_junk_segment(clean_line):
                    continue

                seg_start = i * time_per_line
                seg_end = (i + 1) * time_per_line
                fake_segments.append({
                    "text": clean_line,
                    "start": seg_start,
                    "end": seg_end
                })

        if not fake_segments:
            return []

        # 2. Загружаем модель выравнивания (wav2vec2-based, НЕ whisper)
        log("🔄 Загрузка модели выравнивания...")
        model_a, metadata = whisperx.load_align_model(
            language_code="ru",
            device=device
        )

        # 3. Выравниваем — получаем точные пословные таймкоды
        try:
            aligned_result = whisperx.align(
                fake_segments,
                model_a,
                metadata,
                audio,
                device,
                return_char_alignments=False
            )
        finally:
            # Гарантированная очистка даже при ошибке alignment
            del model_a, metadata
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            del audio
            gc.collect()

        # 4. Извлекаем слова с таймкодами
        all_words = []
        segments = aligned_result.get("segments", [])

        for seg_idx, segment in enumerate(segments):
            words_in_seg = segment.get("words", [])

            if not words_in_seg:
                continue

            for i, word in enumerate(words_in_seg):
                if "start" not in word or "end" not in word:
                    continue

                clean = clean_word(word["word"])
                if not re.match(r'^[а-яё]+$', clean):
                    continue
                if len(clean) < 2:
                    continue

                # Определяем конец строки
                is_end_of_line = (i == len(words_in_seg) - 1)
                if not is_end_of_line and i < len(words_in_seg) - 1:
                    next_word = words_in_seg[i + 1]
                    if "start" in next_word:
                        pause_after = next_word["start"] - word["end"]
                        if pause_after > 0.3:
                            is_end_of_line = True

                w_obj = {
                    "word": word["word"].strip(),
                    "start": word["start"],
                    "end": word["end"],
                    "is_eol": is_end_of_line,
                    "confidence": word.get("score", 0.9)
                }
                all_words.append(w_obj)

        log(f"🎯 Forced Alignment: получено {len(all_words)} слов с таймкодами")
        return all_words

    except Exception as e:
        log(f"⚠️ Forced Alignment failed: {e}")
        traceback.print_exc()
        return []


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

def clean_word(w):
    w = w.lower().replace('ё', 'е')
    return re.sub(r'[^\w]', '', w)

def score_candidates(all_words, extended_range=False):
    """Улучшенная алгоритмическая оценка слов с логированием.

    Включает фонетический анализ рифм: если слово рифмуется с другим
    концом строки — бонус (игрок сможет угадать по рифме).
    """
    if not all_words:
        return []

    scores = []
    total_words = len(all_words)
    freq_map = Counter([clean_word(w['word']) for w in all_words])

    # При extended_range расширяем диапазон поиска
    min_progress = 0.20 if extended_range else 0.30
    max_progress = 0.85 if extended_range else 0.70
    min_audio_pos = MIN_AUDIO_POSITION * 0.7 if extended_range else MIN_AUDIO_POSITION

    # Собираем хвосты всех EOL-слов для проверки рифм
    eol_tails = {}  # {индекс: хвост}
    for i, w in enumerate(all_words):
        if w.get('is_eol', False):
            c = clean_word(w['word'])
            if len(c) >= 3:
                eol_tails[i] = get_russian_syllable_tail(c)

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

        # Валидация таймкодов (баги alignment)
        word_dur = w_obj['end'] - w_obj['start']
        if word_dur < 0.15 or word_dur > 5.0:
            continue
        if i > 0 and w_obj['start'] < all_words[i - 1]['end'] - 0.05:
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

        # Бонус за фонетическую рифму с соседними EOL-словами
        # Если слово рифмуется с другим концом строки — игрок может угадать!
        if w_obj.get('is_eol', False) and len(clean) >= 3:
            my_tail = get_russian_syllable_tail(clean)
            if len(my_tail) >= 2:
                for other_i, other_tail in eol_tails.items():
                    if other_i != i and other_tail == my_tail:
                        score += 15
                        break  # Одного совпадения достаточно

        # Штраф за большой gap перед словом (Whisper мог пропустить слова —
        # контекст будет неполным, плохо для викторины)
        if i > 0:
            gap = w_obj['start'] - all_words[i - 1]['end']
            if gap > 2.0:
                score -= 30  # Сильный штраф — контекст будет с "..."
            elif gap > 1.0:
                score -= 10  # Лёгкий штраф

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

def _find_answer_line_start(all_words, target_idx, answer_line):
    """Находит индекс первого слова строки ответа в all_words.

    Ищет первое слово из answer_line в all_words, двигаясь назад от target_idx.
    Возвращает индекс первого слова строки или -1 если не найдено.
    """
    if not answer_line:
        return -1

    # Извлекаем первое слово строки
    line_words = re.findall(r'[а-яёА-ЯЁa-zA-Z]+', answer_line)
    if not line_words:
        return -1

    first_word_clean = clean_word(line_words[0])
    if len(first_word_clean) < 2:
        # Первое слово слишком короткое (предлог) — берём второе
        if len(line_words) > 1:
            first_word_clean = clean_word(line_words[1])
        else:
            return -1

    # Ищем назад от target_idx (не дальше 20 слов)
    search_start = max(0, target_idx - 20)
    for j in range(target_idx - 1, search_start - 1, -1):
        if clean_word(all_words[j]['word']) == first_word_clean:
            return j

    return -1


def calculate_timings(all_words, target_idx, is_lyrics_mode=False, answer_line=""):
    """Вычисляет тайминги для вопроса и ответа.

    is_lyrics_mode + answer_line: обрезает аудио на границе предыдущей СТРОКИ,
    а не предыдущего слова (чтобы игроки не слышали слова из строки ответа).
    """
    if not all_words or target_idx < 0 or target_idx >= len(all_words):
        return (0, 0), (0, 0)

    target_word = all_words[target_idx]
    target_start_ms = int(target_word['start'] * 1000)

    # Обрезка: всегда по target_start_ms - offset.
    # Это гарантирует что аудио заканчивается прямо перед словом-ответом,
    # даже если Whisper/alignment пропустил слова между prev_word и target.
    CUT_OFFSET_MS = 200  # 200мс до начала слова-ответа
    cut_ms = max(0, target_start_ms - CUT_OFFSET_MS)

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

    q_end_ms = cut_ms
    
    # Финальная проверка: если вдруг фрагмент получился подозрительно коротким
    # (например, из-за сбоя логики или очень раннего слова), форсируем длину.
    if (q_end_ms - q_start_ms) < MIN_DURATION:
        # Пытаемся отступить назад на 26 секунд от ответа
        q_start_ms = max(0, cut_ms - 26000)
        
    # Настройки для ОТВЕТА: 5 сек до ответа + 10-15 сек после (чтобы был слышен ответ)
    a_start_ms = max(0, target_start_ms - 5000)  # 5 секунд до слова-ответа
    a_duration_ms = 15000  # 15 секунд всего (5 сек до + 10 сек после ответа)
    a_end_ms = min(int(all_words[-1]['end']*1000) + 1000, a_start_ms + a_duration_ms)
    
    return (q_start_ms, q_end_ms), (a_start_ms, a_end_ms)

def build_context_string(all_words, target_idx, **kwargs):
    """Контекст = слова прямо перед ответом (последняя фраза).

    Если между последним Whisper-словом и ответом есть gap >0.8с,
    значит Whisper пропустил слова — добавляем '...' чтобы показать
    что аудио продолжается (игрок слышит больше чем видит в тексте).
    """
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

    context = " ".join(current_ctx_words)

    # Если у слова-ответа есть пропущенные lyrics-слова перед ним
    # (Whisper не услышал, но они звучат в аудио) — вставляем их в контекст.
    # Пример: Whisper дал "на этой площадке", lyrics = "на этой дэнс площадке"
    # → skipped_before = ["дэнс"] → контекст = "на этой дэнс ___"
    target_word = all_words[target_idx]
    skipped = target_word.get('skipped_before', [])
    if skipped:
        context += " " + " ".join(skipped)
    else:
        # Нет данных из lyrics — проверяем gap по таймкодам
        if target_idx > 0:
            prev_end = all_words[target_idx - 1]['end']
            target_start = target_word['start']
            gap = target_start - prev_end
            if gap > 0.8:
                context += " ..."

    return context + " ___"

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