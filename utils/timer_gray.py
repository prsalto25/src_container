import time
from datetime import datetime

class TimerGray:
    def __init__(self):
        self.now = datetime.utcnow()
        self.now_t = time.time()
        self.count = 0
        self.dt = 0
        self.last_sent = {}

    def update(self):
        self.now = datetime.utcnow()
        self.count += 1
        self.dt = time.time() - self.now_t
        self.now_t = time.time()

    def has_exceeded(self, name, duration):
        if name not in self.last_sent:
            self.last_sent[name] = self.now_t
            return False
        if self.now_t - self.last_sent[name] > duration:
            self.last_sent[name] = self.now_t
            return True
        return False

    def now_system_time(self):
        return datetime.now()  # Return system time