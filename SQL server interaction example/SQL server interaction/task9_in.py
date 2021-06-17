import sqlite3
import time
from datetime import datetime
import random
import string
import csv
import sys


def access_database_seq(dbfile, query, args):
    connect = sqlite3.connect(dbfile)
    cursor = connect.cursor()
    cursor.execute(query, args)
    connect.commit()
    connect.close()


def access_database_with_result_seq(dbfile, query, args):
    connect = sqlite3.connect(dbfile)
    cursor = connect.cursor()
    rows = cursor.execute(query, args).fetchall()
    connect.commit()
    connect.close()
    return rows


def task9_in(file):
    """ Reads a csv file and updates the database with the session login and logout times of the
    specified users."""
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        data = []
        for i, row in enumerate(csv_reader):
            data.append(row)

    for i in range(len(data)):
        try:
            date_year = int(data[i][1][:4])
            date_month = int(data[i][1][4:6])
            date_day = int(data[i][1][6:8])
            date_hour = int(data[i][1][8:10])
            date_min = int(data[i][1][10:12])
            date_formatted = datetime(
                date_year,
                date_month,
                date_day,
                date_hour,
                date_min).strftime('%Y-%m-%d %H:%M:%S')

        except ValueError:
            return print(
                'Invalid date format in csv, format must be yyyymmddhhmm.')

        username = data[i][0]
        userid = access_database_with_result_seq(
            "traffic_app.db",
            "SELECT userid FROM user_pass WHERE username=?",
            (username,
             ))[0][0]

        if data[i][2] == 'login':
            session_id = ''.join(
                random.choices(
                    string.ascii_uppercase +
                    string.digits,
                    k=10))  
            access_database_seq(
                "traffic_app.db", "INSERT INTO all_sessions(userid, session_magic, session_start)\
                   VALUES (?,?,?)", (userid, session_id, date_formatted))
        elif data[i][2] == 'logout':
            row = access_database_with_result_seq(
                "traffic_app.db",
                "SELECT session_magic FROM all_sessions WHERE userid =?",
                (userid,
                 ))

            # Select session magic of last added entry, assuming that will be
            # the correct entry to log out...
            session_id = row[-1][0]
            check_correct_entry = access_database_with_result_seq(
                "traffic_app.db",
                "SELECT session_end FROM all_sessions WHERE session_magic=?",
                (session_id,
                 ))
            # Should pass if correct entry selected as there will be no logout
            # time...
            if check_correct_entry[0][0] is None:
                access_database_seq(
                    "traffic_app.db", "UPDATE all_sessions SET session_end=? \
                WHERE userid=? AND session_magic=?", (date_formatted, userid, session_id))

            else:
                return print('Error updating database.')

    return print(f'Updated database with data from file: {file}.')


def main():
    if len(sys.argv) <= 1:
        print("No file to read in provided.")
    else:
        task9_in(sys.argv[1])


if (__name__ == "__main__"):
    main()
