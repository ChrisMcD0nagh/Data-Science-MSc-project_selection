import sqlite3
import csv
from datetime import datetime
import random
import string
import sys


def access_database_sec(dbfile, query, args):
    connect = sqlite3.connect(dbfile)
    cursor = connect.cursor()
    cursor.execute(query, args)
    connect.commit()
    connect.close()


def access_database_with_result_sec(dbfile, query, args):
    connect = sqlite3.connect(dbfile)
    cursor = connect.cursor()
    rows = cursor.execute(query, args).fetchall()
    connect.commit()
    connect.close()
    return rows


def task8_in(file):
    """ Reads a csv that updates the database with traffic records."""
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        data = []
        for i, row in enumerate(csv_reader):
            data.append(row)

        userid = 99
        session_id = ''.join(
            random.choices(
                string.ascii_uppercase +
                string.digits,
                k=10))

        added = 0
        removed = 0

    failed = []  # Keeps track of the row numbers of undo entries that failed
    for i in range(len(data)):
        # Format start and end inputs to format in database...
        try:
            date_year = int(data[i][0][:4])
            date_month = int(data[i][0][4:6])
            date_day = int(data[i][0][6:8])
            date_hour = int(data[i][0][8:10])
            date_min = int(data[i][0][10:12])
            date_formatted = datetime(
                date_year,
                date_month,
                date_day,
                date_hour,
                date_min).strftime('%Y-%m-%d %H:%M:%S')

        except (IndexError, ValueError):
            return print(
                'Invalid date format in csv, format must be yyyymmddhhmm.')

        location = data[i][2]
        vec_type = data[i][3]
        occupancy = int(data[i][4])
        if 0 <= occupancy <= 4:
            if data[i][1] == 'add':
                access_database_sec(
                    "traffic_app.db",
                    "INSERT INTO traffic_records(userid, session_magic, location, \
                                        type, occupancy, entry_time) VALUES (?,?,?,?,?,?)",
                    (userid,
                     session_id,
                     location,
                     vec_type,
                     occupancy,
                     date_formatted))
                added += 1

            elif data[i][1] == 'undo':
                check_record_exist = access_database_with_result_sec(
                    "traffic_app.db", "SELECT entry_no FROM traffic_records WHERE \
                            location=? AND type=? AND occupancy=?",
                    (location, vec_type, occupancy))
                # Should always pass as add should already exist...
                if len(check_record_exist) > 0:
                    identical_records = [rec[0] for rec in check_record_exist]
                    identical_records_undone = access_database_with_result_sec("traffic_app.db", "SELECT count(entry_no) \
                                                                        FROM records_to_undo WHERE location = ? \
                                                                                AND type = ? AND occupancy = ?",
                                                                               (location, vec_type, occupancy))
                    if identical_records_undone[0][0] < len(identical_records):
                        index = identical_records_undone[0][0]
                        entry_no = identical_records[index]
                        # Check record to undo entry does not already exist...
                        row = access_database_with_result_sec(
                            "traffic_app.db", "SELECT * FROM records_to_undo WHERE \
                                entry_no=?", (entry_no,))
                        if len(row) == 0:
                            access_database_sec(
                                "traffic_app.db",
                                "INSERT INTO records_to_undo VALUES (?,?,?,?,?,?)",
                                (entry_no,
                                 userid,
                                 session_id,
                                 location,
                                 vec_type,
                                 occupancy))
                            removed += 1
                    else:
                        failed.append(i + 1)
                else:
                    failed.append(i + 1)
            else:
                failed.append(i + 1)
        else:
            failed.append(i + 1)
    if len(failed) == 0:
        return print(
            f'Update complete with data from file: {file}. Added {added} record(s), removed {removed} record(s)')
    else:
        return print(f"""Updated database with data from file: {file}.\n
Note that undos on line(s): {failed} could not be processed as the entries do not exist (or have already been undone) or contain invalid inputs.\n
Updated completed adding {added} record(s) and removed {removed} record(s)""")


def main():
    if len(sys.argv) <= 1:
        print("No file to read in provided.")
    else:
        task8_in(sys.argv[1])


if (__name__ == "__main__"):
    main()
