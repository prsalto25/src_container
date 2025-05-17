import time

class EventTimer:
    def __init__(self, time_expired=3):
        self.timestamps = []  # Store event timestamps
        self.time_expired = time_expired

    def hit(self):
        now = time.time()

        # Check if the last event was too long ago
        if self.timestamps and now - self.timestamps[-1] > self.time_expired:
            self.timestamps.clear()  # Reset if expired

        self.timestamps.append(now)

        # Calculate duration between first and latest event
        if len(self.timestamps) > 1:
            duration = self.timestamps[-1] - self.timestamps[0]
        else:
            duration = 0  # No duration if only one event

        return duration

# Example Usage
if __name__ == "__main__":
    timer = EventTimer()
    while True:
        input("Press Enter to hit the timer...")  # Simulating hits
        duration = timer.hit()
        print(f"Duration between first and latest hit: {duration:.4f} seconds")