import sqlite3
import hashlib

def access_database(dbfile, query):
    connect = sqlite3.connect(dbfile)
    cursor = connect.cursor()
    cursor.execute(query)
    connect.commit()
    connect.close()


def access_database_with_result(dbfile, query):
    connect = sqlite3.connect(dbfile)
    cursor = connect.cursor()
    rows = cursor.execute(query).fetchall()
    connect.commit()
    connect.close()
    return rows

passwords = {'test1': 'password1',
             'test2': 'password2',
             'test3': 'password3',
             'test4': 'password4',
             'test5': 'password5',
             'test6': 'password6',
             'test7': 'password7',
             'test8': 'password8',
             'test9': 'password9',
             'test10': 'password10',
             }
sqltables = {
    'user_pass',
    'current_sessions',
    'all_sessions',
    'traffic_records',
    'records_to_undo'}


def hash_password(password):
    crypt = hashlib.md5()
    crypt.update(str.encode(password))
    return crypt.hexdigest()


def setup_assessment_tables(dbfile):
    # Get rid of any existing data...
    for table in sqltables:
        access_database(dbfile, f"DROP TABLE IF EXISTS {table}")

    # Freshly setup tables...
    access_database(
        dbfile,
        "CREATE TABLE user_pass (userid INTEGER PRIMARY KEY AUTOINCREMENT, \
                    username TEXT, pass TEXT)")
    access_database(
        dbfile,
        "CREATE TABLE current_sessions (userid INTEGER, session_magic TEXT)")
    access_database(
        dbfile, "CREATE TABLE all_sessions (userid INTEGER, session_magic TEXT,\
                    session_start DATETIME, session_end DATETIME)")
    access_database(
        dbfile, "CREATE TABLE traffic_records (entry_no INTEGER PRIMARY KEY AUTOINCREMENT,\
                    userid INTEGER, session_magic TEXT, location TEXT, type TEXT, occupancy INTEGER, entry_time DATETIME)")
    access_database(
        dbfile, "CREATE TABLE records_to_undo (entry_no INTEGER, userid INTEGER,\
                    session_magic TEXT, location TEXT, type TEXT, occupancy INTEGER)")

    # Populate the user_pass table with predefined users and passwords...
    for user, password in passwords.items():
        hashed_pass = hash_password(password)
        access_database(
            dbfile,
            f"INSERT INTO user_pass(username,pass) VALUES ('{str(user)}', '{str(hashed_pass)}')")
    print(f'Created new database: {dbfile}')
setup_assessment_tables("traffic_app.db")
