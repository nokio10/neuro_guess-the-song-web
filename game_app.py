import os
import json
import random
import string
import shutil
import requests
import re
import glob
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, join_room

app = Flask(__name__)
app.config['SECRET_KEY'] = 'production_secret_key_change_me'

# Используем threading + gunicorn gthread/simple-websocket вместо eventlet.
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='threading',
    ping_timeout=60,  # Увеличиваем timeout для медленных соединений
    ping_interval=25,  # Частота ping для проверки соединения
    logger=True,
    engineio_logger=False
)

# --- КОНФИГ ---
BASE_DIR = os.getcwd()
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')
GENERATOR_URL = os.environ.get("GENERATOR_URL", "http://generator:5001")
MIN_PLAYER_ANSWER_LENGTH = 4

if not os.path.exists(MEDIA_ROOT): os.makedirs(MEDIA_ROOT)

def normalize_answer_strict(text):
    if not text: return ""
    text = text.lower().replace('ё', 'е')
    return re.sub(r'[^\w]', '', text)

# --- СОСТОЯНИЕ ИГРЫ ---
def validate_player_answer_text(text):
    normalized = normalize_answer_strict(text)
    if len(normalized) < MIN_PLAYER_ANSWER_LENGTH:
        return False, f'Ответ должен содержать минимум {MIN_PLAYER_ANSWER_LENGTH} символа(ов).'
    return True, ""


class GameState:
    def __init__(self):
        self.game_id = None
        self.is_active = False
        self.current_q_index = -1
        self.questions = []
        self.players = {}
        self.current_phase = 'idle' 
        self.inputs_enabled = False
        self.final_results = None

game = GameState()

def load_game_questions(gid):
    path = os.path.join(MEDIA_ROOT, f"{gid}-questions.json")
    try:
        if not os.path.exists(path): return []
        with open(path, 'r', encoding='utf-8') as f: return json.load(f)
    except: return []

# --- РОУТЫ ---

@app.route('/')
def index(): return render_template('index.html')

@app.route('/admin')
def admin(): return render_template('admin.html')

# --- SOCKETS: ВХОД И УПРАВЛЕНИЕ ---

@socketio.on('join_game')
def on_join(data):
    name = data.get('name', '').strip()

    if not name: return
    if not game.game_id:
        emit('join_error', {'msg': 'Игра ещё не создана!'}, to=request.sid)
        return

    # Логика переподключения
    target_name = name.lower().strip()
    old_sid = None
    prev_score = 0
    prev_answer = None

    for sid, p in game.players.items():
        if p['name'].lower().strip() == target_name:
            prev_score = p['score']
            prev_answer = p.get('last_answer')
            old_sid = sid
            break

    if old_sid: del game.players[old_sid]

    game.players[request.sid] = {
        'name': name,
        'score': prev_score,
        'last_answer': prev_answer  # Сохраняем предыдущий ответ при переподключении
    }

    join_room('players')
    emit('join_success', {'game_id': game.game_id}, to=request.sid)

    # Отправляем полное состояние игры переподключившемуся игроку
    state = _get_client_state()
    emit('game_status', state, to=request.sid)

    # Если идет вопрос (проигрывается трек или фаза ответов) - отправляем данные вопроса
    if game.current_phase in ['question', 'answer'] and game.current_q_index >= 0:
        current_q = game.questions[game.current_q_index]
        emit('new_question', {
            'index': game.current_q_index + 1,
            'total': len(game.questions),
            'question': current_q.get('question', ''),
            'type': current_q.get('type', 'text'),
            'track_meta': current_q.get('track_meta', ''),
        }, to=request.sid)

    # Если идет фаза ответов и у игрока нет ответа - разрешаем отвечать
    if game.current_phase == 'question' and game.inputs_enabled and not prev_answer:
        emit('allow_answers', to=request.sid)

    _broadcast_admin_info()

@socketio.on('admin_create_game')
def admin_create():
    if game.game_id:
        emit('admin_error', {'msg': 'Игра уже создана!'}, to=request.sid)
        return
    game.game_id = ''.join(random.choices(string.ascii_uppercase, k=4))
    game.questions = []
    game.current_q_index = -1
    game.is_active = False
    game.players = {}
    game.current_phase = 'idle'
    socketio.emit('game_reset', to='players') 
    _broadcast_admin_info()

@socketio.on('admin_start_round')
def admin_start_round():
    qs = load_game_questions(game.game_id)
    if not qs:
        emit('admin_error', {'msg': 'Вопросы не найдены!'}, to=request.sid)
        return
    game.questions = qs
    game.is_active = True
    game.current_q_index = -1
    game.current_phase = 'idle'
    _broadcast_admin_info()

@socketio.on('admin_hard_reset')
def admin_hard_reset():
    game.game_id = None
    game.questions = []
    game.players = {}
    game.is_active = False
    game.current_phase = 'idle'
    socketio.emit('game_reset', to='players')
    _broadcast_admin_info()

# --- ГЕНЕРАТОР ---

@socketio.on('gen_start')
def on_gen_start(data):
    if not game.game_id:
        emit('gen_log', {'msg': '❌ Сначала создайте игру!'}, to=request.sid)
        return
    try:
        payload = {'game_id': game.game_id, 'token': data.get('token'), 'urls': data.get('urls', '').split('\n')}
        resp = requests.post(f"{GENERATOR_URL}/start", json=payload, timeout=2)
        if resp.status_code == 200:
            emit('gen_log', {'msg': '🚀 Задача отправлена...'}, to=request.sid)
            socketio.start_background_task(poll_generator_task, request.sid)
        else:
            emit('gen_log', {'msg': f'⚠️ Ошибка генератора: {resp.text}'}, to=request.sid)
    except Exception as e:
        emit('gen_log', {'msg': f'❌ Ошибка связи: {e}'}, to=request.sid)

def poll_generator_task(sid):
    last_log_idx = 0
    fail_count = 0
    while True:
        socketio.sleep(1.5)
        try:
            r = requests.get(f"{GENERATOR_URL}/status", timeout=2).json()
            fail_count = 0 
            if len(r['logs']) > last_log_idx:
                for i in range(last_log_idx, len(r['logs'])):
                    socketio.emit('gen_log', {'msg': r['logs'][i]}, to=sid)
                last_log_idx = len(r['logs'])
            socketio.emit('gen_progress', {'percent': r['progress']}, to=sid)
            if r['status'] == 'finished':
                game.questions = load_game_questions(game.game_id)
                socketio.emit('gen_finished', to=sid)
                _broadcast_admin_info()
                break
        except:
            fail_count += 1
            if fail_count >= 10: break

# --- ГЕЙМПЛЕЙ ---

@socketio.on('admin_next_question')
def admin_next():
    if not game.is_active: return
    game.current_q_index += 1
    if game.current_q_index >= len(game.questions):
        admin_end()
    else:
        game.current_phase = 'question'
        game.inputs_enabled = False
        for pid in game.players: game.players[pid]['last_answer'] = None
        q = game.questions[game.current_q_index]
        
        audio_url = f"{game.game_id}-media/{q['id']}-1.mp3"
        
        socketio.emit('new_question', {
            'type': 'text', 
            'question': q['question'], 
            'index': game.current_q_index + 1,
            'track_meta': q.get('track_meta', ''),
        }, to='players')
        
        # ВАЖНО: Отправляем в admin_room, чтобы играло у админа, даже если нажал игрок
        socketio.emit('play_audio', {'file': audio_url}, to='admin_room')
        
        _broadcast_admin_info()

@socketio.on('admin_repeat_question')
def admin_repeat():
    if game.current_q_index >= 0 and game.current_q_index < len(game.questions):
        q = game.questions[game.current_q_index]
        audio_url = f"{game.game_id}-media/{q['id']}-1.mp3"
        emit('play_audio', {'file': audio_url}, to=request.sid)

@socketio.on('admin_audio_finished')
def admin_audio_finished():
    if not game.is_active: return

    # 1. Если закончился ВОПРОС -> Разрешаем отвечать
    if game.current_phase == 'question':
        game.inputs_enabled = True
        socketio.emit('allow_answers', to='players')
        
    # 2. Если закончился ОТВЕТ -> Разрешаем VIP игроку нажать "Далее"
    elif game.current_phase == 'answer':
        socketio.emit('allow_next_question', to='players')

@socketio.on('submit_answer')
def on_answer(data):
    if game.current_phase == 'question' and game.inputs_enabled:
        answer_text = (data.get('answer') or '').strip()
        is_valid, message = validate_player_answer_text(answer_text)
        if not is_valid:
            emit('answer_validation_error', {
                'msg': message,
                'min_length': MIN_PLAYER_ANSWER_LENGTH,
            }, to=request.sid)
            return
        game.players[request.sid]['last_answer'] = answer_text
        _broadcast_admin_info()
        if len(game.players) > 0 and all(p['last_answer'] for p in game.players.values()):
            game.inputs_enabled = False
            socketio.emit('start_timer', {'seconds': 3}, to='players')
            socketio.start_background_task(_auto_show)

def _auto_show():
    socketio.sleep(3)
    if game.current_phase == 'question':
        with app.app_context(): _reveal()

@socketio.on('admin_show_answer')
def admin_show(): _reveal()

@socketio.on('player_next_question')
def player_next_question():
    # 1. Вычисляем VIP игрока
    if not game.players: return
    vip_sid = list(game.players.keys())[0]
    
    # 2. Проверяем права
    if request.sid == vip_sid:
        admin_next()

def _reveal():
    if game.current_phase != 'question': return
    game.current_phase = 'answer'
    
    q = game.questions[game.current_q_index]
    correct_clean = normalize_answer_strict(q['answer'])
    deltas = {}
    
    # 1. Подсчет очков
    for p in game.players.values():
        ans = p.get('last_answer')
        user_clean = normalize_answer_strict(ans)
        is_correct = (user_clean == correct_clean and len(correct_clean) > 0)
        pts = 2 if is_correct else 0
        p['score'] += pts
        deltas[p['name']] = pts
        
    lb = sorted([{'name': p['name'], 'score': p['score']} for p in game.players.values()], key=lambda x:x['score'], reverse=True)

    # 2. Определение VIP и последнего раунда
    vip_sid = list(game.players.keys())[0] if game.players else None
    is_last_round = (game.current_q_index >= len(game.questions) - 1)

    # 3. Отправка результатов каждому игроку персонально (с my_delta)
    for sid, p in game.players.items():
        socketio.emit('show_answer_client', {
            'answer': q['answer'],
            'track_meta': q.get('track_meta', ''),
            'deltas': deltas,
            'leaderboard': lb,
            'vip_id': vip_sid,
            'is_last': is_last_round,
            'my_delta': deltas.get(p['name'], 0)  # Персональный результат игрока
        }, to=sid)
    
    # 4. Аудио ответа
    audio_url = f"{game.game_id}-media/{q['id']}-2.mp3"
    socketio.emit('play_audio', {'file': audio_url}, to='admin_room')
    
    # 5. ОТПРАВКА ОТВЕТОВ В АДМИНКУ (ЭТО БЫЛО УТЕРЯНО)
    res_list = []
    for p in game.players.values():
        u_cl = normalize_answer_strict(p['last_answer'])
        res_list.append({
            'name': p['name'], 
            'answer': p['last_answer'], 
            'is_correct': (u_cl == correct_clean and len(correct_clean) > 0)
        })
    socketio.emit('round_results', res_list, to='admin_room')
    
    _broadcast_admin_info()

@socketio.on('admin_end_game')
def admin_end():
    game.is_active = False
    game.current_phase = 'finished'
    lb = sorted([{'name': p['name'], 'score': p['score']} for p in game.players.values()], key=lambda x:x['score'], reverse=True)
    wins = [p for p in lb if p['score'] == lb[0]['score']] if lb else []
    game.final_results = {'leaderboard': lb, 'winners': wins}
    socketio.emit('game_over', game.final_results)
    _broadcast_admin_info()

# --- УПРАВЛЕНИЕ ОЧКАМИ ---

@socketio.on('admin_give_point')
def admin_give(data):
    pid = data.get('id')
    if pid in game.players:
        game.players[pid]['score'] += 1
        _broadcast_admin_info()

@socketio.on('admin_take_point')
def admin_take(data):
    pid = data.get('id')
    if pid in game.players:
        game.players[pid]['score'] -= 1
        _broadcast_admin_info()

@socketio.on('admin_remove_player')
def admin_remove_player(data):
    """Удаление игрока администратором"""
    pid = data.get('id')
    if pid in game.players:
        del game.players[pid]
        # Отправляем игроку сообщение о сбросе (выкинет его из игры)
        socketio.emit('game_reset', to=pid)
        _broadcast_admin_info()

# --- СОХРАНЕННЫЕ ИГРЫ ---

def scan_saved_games():
    games_found = []
    json_files = glob.glob(os.path.join(MEDIA_ROOT, "*-questions.json"))
    
    for jf in json_files:
        filename = os.path.basename(jf)
        gid = filename.replace("-questions.json", "")
        try:
            with open(jf, 'r', encoding='utf-8') as f: qs = json.load(f)
        except: qs = []
            
        media_path = os.path.join(MEDIA_ROOT, f"{gid}-media")
        media_exists = os.path.exists(media_path)
        valid_files = 0
        
        if media_exists and qs:
            for q in qs:
                if os.path.exists(os.path.join(media_path, f"{q['id']}-1.mp3")): 
                    valid_files += 1

        total = len(qs)
        is_valid = (total > 0) and (valid_files == total) # Валидация для кнопки

        games_found.append({
            'game_id': gid,
            'questions_count': total,
            'valid_count': valid_files,
            'is_valid': is_valid,
            'timestamp': os.path.getmtime(jf)
        })
    
    games_found.sort(key=lambda x: x['timestamp'], reverse=True)
    return games_found

@socketio.on('admin_get_saved_games')
def on_get_saved_games():
    games_list = scan_saved_games()
    emit('saved_games_list', games_list, to=request.sid)

@socketio.on('admin_load_game')
def on_load_game(data):
    gid = data.get('game_id')
    qs = load_game_questions(gid)
    if not qs:
        emit('admin_error', {'msg': 'Не удалось загрузить вопросы!'}, to=request.sid)
        return

    game.game_id = gid
    game.questions = qs
    game.current_q_index = -1
    game.is_active = False
    game.current_phase = 'idle'
    game.players = {} 
    
    socketio.emit('game_reset', to='players')
    emit('admin_success', {'msg': f'Игра {gid} загружена!'}, to=request.sid)
    _broadcast_admin_info()

@socketio.on('admin_delete_game')
def on_delete_game(data):
    gid = data.get('game_id')
    if not gid: return
    
    if game.game_id == gid:
        admin_hard_reset()
    
    try:
        json_path = os.path.join(MEDIA_ROOT, f"{gid}-questions.json")
        media_path = os.path.join(MEDIA_ROOT, f"{gid}-media")
        if os.path.exists(json_path): os.remove(json_path)
        if os.path.exists(media_path): shutil.rmtree(media_path)
        
        emit('admin_success', {'msg': 'Удалено'}, to=request.sid)
        on_get_saved_games()
    except Exception as e:
        emit('admin_error', {'msg': str(e)}, to=request.sid)

# --- ИНФО ---

def _get_client_state():
    if game.current_phase == 'finished': return {'state': 'finished', 'final_results': game.final_results}
    if not game.is_active: return {'state': 'idle'}
    q = game.questions[game.current_q_index] if 0 <= game.current_q_index < len(game.questions) else None
    return {
        'state': game.current_phase, 
        'inputs_enabled': game.inputs_enabled, 
        'question_data': {
            'question': q['question'],
            'index': game.current_q_index + 1,
            'track_meta': q.get('track_meta', ''),
        } if q else None
    }

def _broadcast_admin_info():
    players_list = []
    for sid, p in game.players.items():
        p_data = p.copy()
        p_data['id'] = sid
        players_list.append(p_data)
    players_list.sort(key=lambda x: x['score'], reverse=True)
    
    qs = [{'id': q['id'], 'status': ('done' if i<game.current_q_index else ('current' if i==game.current_q_index else 'waiting')), 'type': 'text'} for i,q in enumerate(game.questions)]
    
    socketio.emit('admin_update', {
        'game_id': game.game_id, 
        'players': players_list, 
        'questions': qs, 
        'game_active': game.is_active, 
        'phase': game.current_phase
    }, to='admin_room')

@socketio.on('join_admin')
def join_admin():
    join_room('admin_room')
    _broadcast_admin_info()

@socketio.on('disconnect')
def on_disconnect():
    """Обработка отключения - помечаем игрока как офлайн, но не удаляем сразу"""
    # НЕ удаляем игрока из game.players, чтобы при переподключении он мог восстановиться
    # Игрок будет автоматически заменен при следующем подключении с тем же именем
    pass

if __name__ == '__main__':
    socketio.run(
        app,
        host='0.0.0.0',
        port=5000,
        debug=False,
        use_reloader=False,
        allow_unsafe_werkzeug=True,
    )
