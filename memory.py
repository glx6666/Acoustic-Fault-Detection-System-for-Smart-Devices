# memory.py
import redis
import json
import os

class RedisMemory:
    """
    Minimal Redis-backed chat memory. Uses a list per session.
    Each item stored as JSON: {"role": "user"|"assistant", "content": "..."}
    """

    def __init__(self, session_id: str = "default", host=None, port=None, db=None):
        host = host or os.environ.get("REDIS_HOST", "localhost")
        port = int(port or os.environ.get("REDIS_PORT", 6379))
        db = int(db or os.environ.get("REDIS_DB", 0))
        self.r = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.key = f"chat:{session_id}"

    def append(self, role: str, content: str):
        msg = {"role": role, "content": content}
        self.r.rpush(self.key, json.dumps(msg))

    def get_history(self, limit: int = 20):
        # return last `limit` messages as list of dicts
        msgs = self.r.lrange(self.key, -limit, -1)
        return [json.loads(m) for m in msgs]

    def clear(self):
        self.r.delete(self.key)
