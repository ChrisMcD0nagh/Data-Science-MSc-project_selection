import sqlite3
import csv
from datetime import datetime
import sys
from collections import defaultdict


def access_database_with_result(dbfile, query):
    connect = sqlite3.connect(dbfile)
    cursor = connect.cursor()
    rows = cursor.execute(query).fetchall()
    connect.commit()
    connect.close()
    return rows


def access_database_with_result_sec(dbfile, query, args):
    connect = sqlite3.connect(dbfile)
    cursor = connect.cursor()
    rows = cursor.execute(query, args).fetchall()
    connect.commit()
    connect.close()
    return rows


def task8_out(start, end):
    """ Returns traffic records between the start and end date provided to a csv."""
    # Format start and end inputs in form yyyymmddhhmm...
    try:
        start_year = int(start[:4])
        start_month = int(start[4:6])
        start_day = int(start[6:8])
        start_hour = int(start[8:10])
        start_min = int(start[10:12])
        start_formatted = datetime(
            start_year,
            start_month,
            start_day,
            start_hour,
            start_min).strftime('%Y-%m-%d %H:%M:%S')

        end_year = int(end[:4])
        end_month = int(end[4:6])
        end_day = int(end[6:8])
        end_hour = int(end[8:10])
        end_min = int(end[10:12])
        end_formatted = datetime(
            end_year,
            end_month,
            end_day,
            end_hour,
            end_min).strftime('%Y-%m-%d %H:%M:%S')

    except ValueError:
        return print(
            'Invalid date format in start or end date, format must be yyyymmddhhmm.')

    # Pull relevant records from the database...
    all_records = access_database_with_result_sec(
        "traffic_app.db", "SELECT entry_no, userid, \
    session_magic, location, type, occupancy FROM traffic_records WHERE \
                                entry_time >= ? AND  entry_time <= ? \
                                ORDER BY entry_no", (start_formatted, end_formatted))

    to_ignore = access_database_with_result(
        "traffic_app.db", "SELECT * FROM records_to_undo")
    #print(f'All records: {all_records}')
    #print(f'to_ignore: {to_ignore}')
    ret_unformatted = []
    [ret_unformatted.append(record)
     for record in all_records if record not in to_ignore]
    
    # Sum entries with identical locaion and type...
    entry_totals = defaultdict(dict)
    for record in ret_unformatted:
        location = record[3]
        vec_type = record[4]
        occupancy = record[5]
        if vec_type not in entry_totals[location].keys():
            entry_totals[location][vec_type] = [0,0,0,0]
            entry_totals[location][vec_type][occupancy-1] = 1
        else:
            entry_totals[location][vec_type][occupancy-1] += 1

    # Reformat to required output...
    ret = []
    for loc in entry_totals.keys():
        for vec_type, count in entry_totals[loc].items():
            ret.append([loc, vec_type, count[0], count[1], count[2], count[3]])

    with open('task8_out.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in ret:
            csvwriter.writerow(row)
    return print('File read to task8_out.csv')


def main():
    if len(sys.argv) <= 2:
        print("No enough date inputs, expecting a start and an end date in format yyyymmddhhmm.")
    else:
        task8_out(sys.argv[1], sys.argv[2])


if (__name__ == "__main__"):
    main()
