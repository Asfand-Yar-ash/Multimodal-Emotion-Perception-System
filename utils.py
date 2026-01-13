import json
from datetime import datetime, timezone
import threading

def ts_now():
    return datetime.now(timezone.utc).isoformat()

def write_json_line(path, obj):
    line = json.dumps(obj, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return line

# A simple thread-safe store for the latest audio emotion result
class LatestAudio:
    def __init__(self):
        self.lock = threading.Lock()
        self.value = None

    def set(self, val):
        with self.lock:
            self.value = val

    def get(self):
        with self.lock:
            return self.value
