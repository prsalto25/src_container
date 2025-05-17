import MySQLdb
import traceback
import time

class MySQLHelper:
    def __init__(self, host, port, username, password, database, table=""):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database_name = database
        self.table = table
        self.tables = {}
        self.connect()

    def connect(self):
        self.database = MySQLdb.connect(host=self.host, port=self.port, user=self.username, passwd=self.password, db=self.database_name)
        self.cursor = self.database.cursor()

    def get_columns_info(self, columns):
        fields = [field for field, _ in columns]
        field_defs = [f"{field} {_type}" for field, _type in columns]
        return ','.join(fields), ','.join(field_defs)

    def add_table(self, table, columns, drop=False):
        self.tables[table] = []
        fields, field_defs = self.get_columns_info(columns)
        if drop:
            self.cursor.execute(f"DROP TABLE IF EXISTS {self.database_name}.{table}")
        try:
            cmd = f"CREATE TABLE IF NOT EXISTS {self.database_name}.{table} ({field_defs})"
            self.cursor.execute(cmd)
        except Exception as e:
            raise ValueError(f"MySQL Error: {cmd}\n{e}")

    def insert(self, values):
        values_str = ','.join([f'"{v}"' for v in values])
        cmd = f"INSERT INTO {self.database_name}.{self.table} VALUES ({values_str})"
        self.run(cmd)

    def insert_fast(self, table, values):
        values_str = ','.join([f'"{v}"' for v in values])
        self.tables[table].append(f'({values_str})')

    def commit_insert(self, table):
        if self.tables[table]:
            values = ','.join(self.tables[table])
            cmd = f"INSERT INTO {self.database_name}.{table} VALUES {values}"
            self.run(cmd)
            self.tables[table] = []

    def commit_all(self):
        for table in self.tables:
            self.commit_insert(table)

    def run_fetch(self, cmd):
        try:
            self.cursor.execute(cmd)
            self.database.commit()
            return self.cursor.fetchall()
        except Exception:
            print(f"ERROR: {cmd}\n{traceback.format_exc()}")
            time.sleep(1)
            self.reconnect()
            self.cursor.execute(cmd)
            self.database.commit()
            return self.cursor.fetchall()

    def run(self, cmd):
        try:
            self.cursor.execute(cmd)
            self.database.commit()
        except Exception:
            print(f"ERROR: {cmd}\n{traceback.format_exc()}")
            time.sleep(1)
            self.reconnect()
            self.cursor.execute(cmd)
            self.database.commit()

    def reconnect(self):
        self.close()
        self.connect()

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.database:
            self.database.close()


if __name__ == '__main__':
    mysql = MySQLHelper(
        host='127.0.0.1',
        port=3306,
        username='graymatics',
        password='graymatics',
        database='test0',
        table='test1'
    )
    t0 = time.time()
    result = mysql.run_fetch("SELECT * FROM test1")
    print(result)
    print("Elapsed time:", time.time() - t0)
    mysql.close()