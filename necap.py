#!/usr/bin/env python3
# Educational Cybersecurity measures purposes: sanitized for safe sharing, review, and classroom-style inspection of the code here.

import os
import sys
import time
import json
import threading
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, BASE_DIR)

try:
    from yidun_proxyless import *
    import yidun_proxyless as solver
    SOLVER_AVAILABLE = True
    print("✅ CN31 Solver loaded successfully")
    for f in ('dun163.js', 'net.pkl'):
        if os.path.exists(os.path.join(BASE_DIR, f)):
            sz = os.path.getsize(os.path.join(BASE_DIR, f))
            print(f"✅ {f} found ({sz} bytes)")
        else:
            print(f"❌ {f} NOT found")
            SOLVER_AVAILABLE = False
except ImportError as e:
    SOLVER_AVAILABLE = False
    print(f"❌ CN31 Solver not available: {e}")
    import traceback
    traceback.print_exc()

solver_running = False
solver_thread = None
generation_stats = {
    "status": "idle",
    "tokens_generated": 0,
    "start_time": None,
    "threads": 0,
}

# ======================== TOKEN CACHE ========================
# Each token has its own TTL. No auto-purge and no manual flush.
TOKEN_TTL_SECONDS = int(os.environ.get('TOKEN_TTL_SECONDS', 14 * 60))
AUTO_RESTART_SECONDS = int(os.environ.get('AUTO_RESTART_SECONDS', 2 * 60 * 60))  # 2 hours

tokens_cache = []
_seen_tokens = set()
_token_times = {}
token_lock = threading.Lock()
TOKEN_FILE = os.path.join(BASE_DIR, 'validated_tokens.txt')


def _expire_front():
    """Lazily drop expired tokens from the front of the queue when serving / inspecting."""
    global tokens_cache
    now = time.time()
    dropped = []
    with token_lock:
        for t in list(tokens_cache):
            born = _token_times.get(t)
            if born is None or (now - born) > TOKEN_TTL_SECONDS:
                dropped.append(t)
            else:
                break
        if dropped:
            expired_set = set(dropped)
            tokens_cache = [t for t in tokens_cache if t not in expired_set]
            for t in dropped:
                _token_times.pop(t, None)
    if dropped:
        print(f"⏳ Expired {len(dropped)} token(s) (TTL {TOKEN_TTL_SECONDS}s). Queue: {len(tokens_cache)}")
    return len(dropped)


def read_tokens_from_file():
    try:
        if os.path.exists(TOKEN_FILE):
            with open(TOKEN_FILE, 'r') as f:
                return [line.strip() for line in f if line.strip()]
        return []
    except Exception as e:
        print(f"Error reading tokens: {e}")
        return []


def get_new_tokens():
    global tokens_cache
    try:
        file_tokens = read_tokens_from_file()
        new_tokens = [t for t in file_tokens if t not in _seen_tokens]
        if new_tokens:
            now = time.time()
            with token_lock:
                for t in new_tokens:
                    _seen_tokens.add(t)
                    _token_times[t] = now
                    tokens_cache.append(t)
                generation_stats["tokens_generated"] = len(tokens_cache)
            print(f"✅ Added {len(new_tokens)} new tokens. Total: {len(tokens_cache)}")
        return new_tokens
    except Exception as e:
        print(f"Error getting new tokens: {e}")
        return []


# ======================== WORKER TRACKING ========================
WORKER_ACTIVE_WINDOW = int(os.environ.get("WORKER_ACTIVE_WINDOW", 120))
_workers = {}


def _active_workers(now=None):
    global _workers
    now = now if now is not None else time.time()
    _workers = {w: t for w, t in _workers.items() if (now - t) <= WORKER_ACTIVE_WINDOW}
    return len(_workers)


# ======================== SOLVER WORKER ========================
def _clear_solver_cache():
    global tokens_cache, _seen_tokens, _token_times
    with token_lock:
        tokens_cache.clear()
    _seen_tokens.clear()
    _token_times.clear()
    if os.path.exists(TOKEN_FILE):
        try:
            os.remove(TOKEN_FILE)
            print("🗑️ Cleared validated_tokens.txt")
        except Exception as e:
            print(f"Error clearing token file: {e}")


def run_solver_worker(threads=3):
    global solver_running, generation_stats
    print(f"🚀 Starting CN31 solver with {threads} threads...")
    generation_stats["status"] = "running"
    generation_stats["start_time"] = datetime.now().isoformat()
    generation_stats["threads"] = threads
    solver.NUM_THREADS = threads
    try:
        solver.main()
    except KeyboardInterrupt:
        print("⏹️ Solver stopped by user")
    except Exception as e:
        print(f"❌ Solver error: {e}")
        generation_stats["status"] = "error"
        generation_stats["error"] = str(e)
    finally:
        solver_running = False
        generation_stats["status"] = "stopped"


# ======================== AUTO-RESTART (2 h) ========================
def _auto_restart_loop():
    """Every AUTO_RESTART_SECONDS stop solver, clear all cache, restart fresh."""
    while True:
        time.sleep(AUTO_RESTART_SECONDS)
        global solver_running, solver_thread, generation_stats
        print(f"\n⏰ Auto-restart triggered — clearing cache and restarting solver\n")
        _clear_solver_cache()
        if solver_running:
            solver_running = False
            generation_stats["status"] = "stopping"
            time.sleep(2)
        if SOLVER_AVAILABLE:
            try:
                model = initialize_global_model()
                if model is None:
                    print("❌ Auto-restart: model init failed")
                    continue
            except Exception as e:
                print(f"❌ Auto-restart model error: {e}")
                continue
            solver_running = True
            generation_stats["status"] = "restarting"
            solver_thread = threading.Thread(
                target=run_solver_worker,
                args=(generation_stats.get("threads", 3),),
                daemon=True,
            )
            solver_thread.start()


_restart_thread = threading.Thread(target=_auto_restart_loop, daemon=True)
_restart_thread.start()



# ======================== STATUS / HEALTH ========================
@app.route("/stats")
@app.route("/api/status")
def stats():
    _expire_front()
    elapsed = time.time() - generation_stats.get("_boot_time", time.time())
    if elapsed <= 0:
        elapsed = 1

    rate = generation_stats["tokens_generated"] / (elapsed / 60) if elapsed > 0 else 0
    oldest_age = 0
    with token_lock:
        if _token_times:
            oldest_age = round(time.time() - min(_token_times.values()), 1)

    return jsonify({
        "queue_size": len(tokens_cache),
        "total_received": generation_stats["tokens_generated"],
        "total_served": 0,
        "total_expired": 0,
        "total_duplicates": 0,
        "peak_queue": generation_stats.get("peak_queue", len(tokens_cache)),
        "workers_active": _active_workers(),
        "uptime_seconds": round(elapsed, 1),
        "tokens_per_minute": round(rate, 2),
        "token_ttl_seconds": TOKEN_TTL_SECONDS,
        "auto_restart_seconds": AUTO_RESTART_SECONDS,
        "oldest_age_seconds": oldest_age,
        "last_received": None,
        "last_served": None,
        "recent_tokens": [],
        "solver_status": generation_stats["status"],
        "threads": generation_stats.get("threads", 0),
    })


@app.route("/health")
def health():
    _expire_front()
    return jsonify({
        "ok": True,
        "solver_available": SOLVER_AVAILABLE,
        "status": generation_stats["status"],
        "tokens_available": len(tokens_cache),
        "token_ttl_seconds": TOKEN_TTL_SECONDS,
        "auto_restart_seconds": AUTO_RESTART_SECONDS,
        "files": {
            "yidun_proxyless.py": os.path.exists(os.path.join(BASE_DIR, 'yidun_proxyless.py')),
            "dun163.js": os.path.exists(os.path.join(BASE_DIR, 'dun163.js')),
            "net.pkl": os.path.exists(os.path.join(BASE_DIR, 'net.pkl')),
        },
    })


# ======================== START / STOP ========================
@app.route('/start', methods=['POST'])
def start_solver():
    global solver_running, solver_thread, generation_stats

    if solver_running:
        return jsonify({"error": "Solver already running"}), 400

    if not SOLVER_AVAILABLE:
        return jsonify({"error": "CN31 Solver not available"}), 500

    required_files = ['yidun_proxyless.py', 'dun163.js', 'net.pkl']
    missing = [f for f in required_files if not os.path.exists(os.path.join(BASE_DIR, f))]
    if missing:
        return jsonify({"error": f"Missing files: {missing}"}), 500

    data = request.json or {}
    threads = min(data.get("threads", 3), 10)

    try:
        model = initialize_global_model()
        if model is None:
            return jsonify({"error": "Failed to load model (check /debug/model)"}), 500
    except Exception as e:
        return jsonify({"error": f"Model error: {str(e)}"}), 500

    solver_running = True
    generation_stats["status"] = "starting"
    generation_stats["threads"] = threads

    solver_thread = threading.Thread(
        target=run_solver_worker,
        args=(threads,),
        daemon=True,
    )
    solver_thread.start()

    return jsonify({
        "message": "CN31 Solver started",
        "threads": threads,
        "auto_restart_hours": AUTO_RESTART_SECONDS / 3600,
        "status": "running",
    })


@app.route('/stop', methods=['POST'])
def stop_solver():
    global solver_running
    solver_running = False
    generation_stats["status"] = "stopping"
    return jsonify({
        "message": "Stop signal sent",
        "tokens_generated": generation_stats["tokens_generated"],
    })


# ======================== TOKEN API ========================
@app.route("/get-token")
@app.route("/api/get-token")
def get_token():
    get_new_tokens()
    with token_lock:
        if tokens_cache:
            token = tokens_cache.pop(0)
            _token_times.pop(token, None)
            return jsonify({
                "token": token,
                "remaining": len(tokens_cache),
                "ttl_seconds": TOKEN_TTL_SECONDS,
            })
    return jsonify({"error": "No tokens available", "remaining": 0}), 404


@app.route("/tokens/count")
@app.route("/api/tokens/count")
def token_count():
    _expire_front()
    return jsonify({
        "queue_size": len(tokens_cache),
        "total_received": generation_stats["tokens_generated"],
    })


# ======================== DEBUG ========================
@app.route('/debug/files')
def debug_files():
    files = {
        'yidun_proxyless.py': os.path.exists(os.path.join(BASE_DIR, 'yidun_proxyless.py')),
        'dun163.js': os.path.exists(os.path.join(BASE_DIR, 'dun163.js')),
        'net.pkl': os.path.exists(os.path.join(BASE_DIR, 'net.pkl')),
        'validated_tokens.txt': os.path.exists(TOKEN_FILE),
    }
    sizes = {}
    for f in files:
        if files[f]:
            try:
                sizes[f] = os.path.getsize(os.path.join(BASE_DIR, f))
            except Exception:
                sizes[f] = 'error'
    return jsonify({'files': files, 'sizes': sizes, 'cwd': os.getcwd()})


@app.route('/debug/model')
def debug_model():
    try:
        import torch
        if not os.path.exists(os.path.join(BASE_DIR, 'net.pkl')):
            return jsonify({'error': 'net.pkl not found'})
        model = torch.load(os.path.join(BASE_DIR, 'net.pkl'), map_location='cpu')
        return jsonify({
            'model_loaded': True,
            'model_keys': list(model.keys()) if hasattr(model, 'keys') else 'N/A',
            'model_size': os.path.getsize(os.path.join(BASE_DIR, 'net.pkl')),
        })
    except Exception as e:
        return jsonify({'error': str(e), 'model_loaded': False})


# ======================== MAIN ========================
if __name__ == "__main__":
    port = int(os.environ.get('PORT', 6000))
    generation_stats["_boot_time"] = time.time()

    print(f"""
🔐 CN31 Solver - Railway Edition v3
──────────────────────────────────────
Port           : {port}
Solver         : {'✅ Available' if SOLVER_AVAILABLE else '❌ Not Available'}
Token TTL      : {TOKEN_TTL_SECONDS}s ({TOKEN_TTL_SECONDS // 60} min) per-token expiry
Auto-restart   : every {AUTO_RESTART_SECONDS // 3600}h ({AUTO_RESTART_SECONDS}s)
No auto-purge  : tokens expire lazily on serve
No flush       : cache is cleared only on auto-restart
Files:
  - yidun_proxyless.py : {'✅' if os.path.exists(os.path.join(BASE_DIR, 'yidun_proxyless.py')) else '❌'}
  - dun163.js          : {'✅' if os.path.exists(os.path.join(BASE_DIR, 'dun163.js')) else '❌'}
  - net.pkl            : {'✅' if os.path.exists(os.path.join(BASE_DIR, 'net.pkl')) else '❌'}

Routes:
  POST /start                start solver
  POST /stop                 stop solver
  GET  /get-token            dispense one token
  GET  /stats                full stats
  GET  /health               health check
  GET  /                     dashboard
""")

    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
