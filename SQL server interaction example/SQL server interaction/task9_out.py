import sqlite3
import datetime
import calendar
from collections import defaultdict
import csv
import sys


def access_database_with_result_seq(dbfile, query, args):
    connect = sqlite3.connect(dbfile)
    cursor = connect.cursor()
    rows = cursor.execute(query, args).fetchall()
    connect.commit()
    connect.close()
    return rows


def add_months(sourcedate, months):
    month = sourcedate.month - 1 + months
    year = sourcedate.year + month // 12
    month = month % 12 + 1
    day = min(sourcedate.day, calendar.monthrange(year, month)[1])
    return datetime.date(year, month, day)


def task9_out(date):
    """ Returns the hours worked by each user in the date provided, the week ending on the date provided and
    the month ending on the date provided, to a csv. Users who have not worked any hours will not be reported."""
    try:
        date_year = int(date[:4])
        date_month = int(date[4:6])
        date_day = int(date[6:8])
        date_given = datetime.datetime(
            date_year, date_month, date_day)
        date_end_day = (
            date_given +
            datetime.timedelta(
                days=1)).strftime('%Y-%m-%d %H:%M:%S')
        date_start_week = (
            date_given +
            datetime.timedelta(
                days=-
                6)).strftime('%Y-%m-%d %H:%M:%S')
        date_start_month = (add_months(
            date_given, -1) + datetime.timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
        date_given = date_given.strftime('%Y-%m-%d %H:%M:%S')
    except ValueError:
        return print('Invalid date format, must be yyyymmdd.')

    hours_day = access_database_with_result_seq(
        "traffic_app.db",
        "SELECT userid, session_start, session_end \
                                    FROM all_sessions WHERE session_start >= ? AND session_end <= ?",
        (date_given,
         date_end_day))
    hours_week = access_database_with_result_seq(
        "traffic_app.db",
        "SELECT userid, session_start, session_end \
                                    FROM all_sessions WHERE session_start >= ? AND session_end <= ?",
        (date_start_week,
         date_end_day))
    hours_month = access_database_with_result_seq(
        "traffic_app.db",
        "SELECT userid, session_start, session_end \
                                    FROM all_sessions WHERE session_start >= ? AND session_end <= ?",
        (date_start_month,
         date_end_day))
    # Needed to clear dictionarys each time...
    daily_hours = {}
    weekly_hours = {}
    monthly_hours = {}

    daily_hours = defaultdict(lambda: 0, daily_hours)
    for user, start, end in hours_day:
        if user not in daily_hours.keys():
            mins = (datetime.datetime.strptime(end,
                                               '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(start,
                                                                                                 '%Y-%m-%d %H:%M:%S')).seconds / (60 * 60)
            daily_hours[user] = float(f"{mins:.1f}")
        else:
            mins = (datetime.datetime.strptime(end,
                                               '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(start,
                                                                                                 '%Y-%m-%d %H:%M:%S')).seconds / (60 * 60)
            daily_hours[user] += float(f"{mins:.1f}")

    weekly_hours = defaultdict(lambda: 0, weekly_hours)
    for user, start, end in hours_week:
        if user not in weekly_hours.keys():
            mins = (datetime.datetime.strptime(end,
                                               '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(start,
                                                                                                 '%Y-%m-%d %H:%M:%S')).seconds / (60 * 60)
            weekly_hours[user] = float(f"{mins:.1f}")
        else:
            mins = (datetime.datetime.strptime(end,
                                               '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(start,
                                                                                                 '%Y-%m-%d %H:%M:%S')).seconds / (60 * 60)
            weekly_hours[user] += float(f"{mins:.1f}")

    monthly_hours = defaultdict(lambda: 0, monthly_hours)
    for user, start, end in hours_month:
        if user not in monthly_hours.keys():
            mins = (datetime.datetime.strptime(end,
                                               '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(start,
                                                                                                 '%Y-%m-%d %H:%M:%S')).seconds / (60 * 60)
            monthly_hours[user] = float(f"{mins:.1f}")
        else:
            mins = (datetime.datetime.strptime(end,
                                               '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(start,
                                                                                                 '%Y-%m-%d %H:%M:%S')).seconds / (60 * 60)
            monthly_hours[user] += float(f"{mins:.1f}")
    ret_all = []
    [ret_all.append([f'test{i+1}',
                     daily_hours[i + 1],
                     weekly_hours[i + 1],
                     monthly_hours[i + 1]]) for i in range(10)]

    # Remove 0 rows (perhaps not needed)...
    ret = []
    [ret.append(row) for row in ret_all if sum(row[1:]) != 0]

    with open('task9_out.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        for row in ret:
            csvwriter.writerow(row)
    return print('File read to task9_out.csv')


def main():
    if len(sys.argv) <= 1:
        print("No date provided.")
    else:
        task9_out(sys.argv[1])


if __name__ == "__main__":
    main()
