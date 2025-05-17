import MySQLdb
import traceback
import time

class Mysql:
    def __init__(self, args):
        self.ip = args["ip"]
        self.user = args["user"]
        self.passwd = args["pwd"]
        self.db = args['db']
        self.table = args["table"]
        self.tables = {}

        self.connect()

    def connect(self):
        self.database = MySQLdb.connect(self.ip, self.user, self.passwd, self.db)
        self.cursor = self.database.cursor()

    def get_columns_info(self, columns):
        fields = []
        field_and_type = []

        for field, _type in columns:
            fields.append(field)
            field_and_type.append(f'{field} {_type}')

        fields = ','.join(fields)
        field_and_type = (','.join(field_and_type))
        return fields, field_and_type

    # -------------------------------------------------------

    def add_table(self, table, columns, drop=False):
        self.tables[table] = [] 
        self.fields, self.field_and_type = self.get_columns_info(columns)
        if drop:
            self.cursor.execute(f'drop table if exists {self.db}.{table}')
        try:
            cmd = f'create table if not exists {self.db}.{table} ({self.field_and_type})'
            self.cursor.execute(cmd)
        except:
            raise ValueError(f'MYSQL ERROR: {cmd}')

    #def set_table(self, table):
    #    self.table = table

    def insert(self, values): # 0.05s for each insert, 99% of time used by commit
        values = [f'"{v}"' for v in values]
        values = ','.join(values)
        #cmd = f'insert into {self.db}.{self.table} ({self.fields}) values ({values})'
        cmd = f'insert into {self.db}.{self.table} values ({values})'
        self.run(cmd)

    def insert_fast(self, table, values): # may cause error if used together with alter table
        values = [f'"{v}"' for v in values]
        values = ','.join(values)
        self.tables[table].append(f'({values})')

    def commit_insert(self, table):
        if self.tables[table]:
            values = ','.join(self.tables[table])
            cmd = f'insert into {self.db}.{table} values {values}'
            self.run(cmd)
            self.tables[table] = []

    def commit_all(self):
        for table in self.tables:
            self.commit_insert(table)

    #def fetch(self, fields, wherein=None):
    #    fields = ','.join(fields)
    #    cmd = f'select {fields} from {self.db}.{self.table}'
    #    if wherein:
    #        field, values = wherein
    #        values = [f'"{v}"' for v in values]
    #        values = ','.join(values)
    #        cmd += f' where {field} in ({values})'
    #    #print(cmd)
    #    self.run(cmd)
    #    result = self.cursor.fetchall()
    #    return result

    def run_fetch(self, cmd):
        try:
            self.cursor.execute(cmd)
            self.database.commit()
            result = self.cursor.fetchall()
            return result
        except Exception:
            #print(traceback.format_exc())
            print(f'ERROR:{cmd}')
            time.sleep(1)
            self.close()
            self.connect()
            self.cursor.execute(cmd)
            self.database.commit()
            return result

    def run(self, cmd):
        try:
            self.cursor.execute(cmd)
            self.database.commit()
        except Exception:
            print(traceback.format_exc())
            print(cmd)
            time.sleep(1)
            self.close()
            self.connect()
            self.cursor.execute(cmd)
            self.database.commit()

    def close(self):
        self.cursor.close()
        self.database.close()


if __name__ == '__main__':
    mysql_args = {
        "ip":'127.0.0.1',
        "user":'graymatics',
        "pwd":'graymatics',
        "db":'test0',
        "table":"test1",
        #"column": [
        #    #['c1','INT NOT NULL AUTO_INCREMENT, primary key(c1)'],
        #    ['c1','INT NOT NULL'],
        #    ['c2','varchar(40)']
        #    ]
        }

    mysql = Mysql(mysql_args)
    #for i in range(100):
    #    mysql.insert_fast([12345, 'abcde'])
    #mysql.commit_insert()
    t0 = time.time()
    for i in range(1):
        print(mysql.fetch(['c2,c1']))
    print(time.time() -t0)
