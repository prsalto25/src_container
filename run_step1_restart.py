import time
import subprocess
from datetime import datetime, timedelta

# from utils.mysql2 import Mysql
from utils.mysql_helper import MySQLHelper

import config as p

SCRIPT = "run_step2_sql.py"

print(f"Using MYSQL: {p.MYSQL_DATABASE} {p.MYSQL_HOST}:{p.MYSQL_PORT}")

class Update:
    def __init__(self):
        self.last_update = datetime.now() - timedelta(weeks=100)
        self.mysql = MySQLHelper(
            host=p.MYSQL_HOST,
            port=p.MYSQL_PORT,
            username=p.MYSQL_USERNAME,
            password=p.MYSQL_PASSWORD,
            database=p.MYSQL_DATABASE
        )

    def need_restart(self):
        query = "SELECT updatedAt FROM relations"
        results = self.mysql.run_fetch(query)
        for (updatedAt,) in results:
            if updatedAt > self.last_update:
                self.last_update = updatedAt
                return True
        return False

# print("Running: yolo_ultra/run_zmq.sh")
# subprocess.Popen(["bash", "./yolo_ultra/run_zmq.sh"], shell=True)

print("Running: run_step1_restart.py")

update_checker = Update()
while True:
    if update_checker.need_restart():
        print("Restarting...")
        subprocess.run(f'pkill -f {SCRIPT} -9', shell=True)
        subprocess.Popen(f'python3 {SCRIPT}', shell=True)
    time.sleep(p.CHECK_RESTART_DURATION)
