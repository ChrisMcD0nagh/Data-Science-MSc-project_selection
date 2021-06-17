#!/usr/bin/env python

# This is a simple web server for a traffic counting application.
# It's your job to extend it by adding the backend functionality to support
# recording the traffic in a SQL database. You will also need to support
# some predefined users and access/session control. You should only
# need to extend this file. The client side code (html, javascript and css)
# is complete and does not require editing or detailed understanding.

# import the various libraries needed
import http.cookies as Cookie  # some cookie handling support
# the heavy lifting of the web server
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib  # some url parsing support
import base64  # some encoding support
import sqlite3
import hashlib
import random
import string
import time

# Testing commits...
# Access database functions...
# Access database functions...


def access_database(dbfile, query):
    """ Access SQL database with no additional parameters without returning result.
    """
    connect = sqlite3.connect(dbfile)
    cursor = connect.cursor()
    cursor.execute(query)
    connect.commit()
    connect.close()


def access_database_with_result(dbfile, query):
    """ Access SQL database with no additional parameters and return result.
    """
    connect = sqlite3.connect(dbfile)
    cursor = connect.cursor()
    rows = cursor.execute(query).fetchall()
    connect.commit()
    connect.close()
    return rows


def access_database_sec(dbfile, query, args):
    """ Access SQL database with additional parameters without returning result.
    """
    connect = sqlite3.connect(dbfile)
    cursor = connect.cursor()
    cursor.execute(query, args)
    connect.commit()
    connect.close()


def access_database_with_result_sec(dbfile, query, args):
    """ Access SQL database with additional parameters and return the result.
    """
    connect = sqlite3.connect(dbfile)
    cursor = connect.cursor()
    rows = cursor.execute(query, args).fetchall()
    connect.commit()
    connect.close()
    return rows


def hash_password(password):
    """ Hash a string with md5 crypotgraphy.
    """
    crypt = hashlib.md5()
    crypt.update(str.encode(password))
    return crypt.hexdigest()


def check_location_input(location):
    """ Check location input is valid
    """
    punctuation = string.punctuation
    transform = str.maketrans('', '', punctuation)
    cleaned_loc = location.strip().translate(transform)
    return len(cleaned_loc) > 0


def build_response_refill(where, what):
    """ This function builds a refill action that allows part of the
    currently loaded page to be replaced.
    """
    text = "<action>\n"
    text += "<type>refill</type>\n"
    text += "<where>" + where + "</where>\n"
    m = base64.b64encode(bytes(what, 'ascii'))
    text += "<what>" + str(m, 'ascii') + "</what>\n"
    text += "</action>\n"
    return text


# If this action is used, only one instance of it should
# contained in the response and there should be no refill action.
def build_response_redirect(where):
    """This function builds the page redirection action, it indicates which page
    the client should fetch."""
    text = "<action>\n"
    text += "<type>redirect</type>\n"
    text += "<where>" + where + "</where>\n"
    text += "</action>\n"
    return text


# Decide if the combination of user and magic is valid
def handle_validate(iuser, imagic):
    """ Check the user/magic combination exists in the current_sessions table.
    """
    row = access_database_with_result_sec(
        "traffic_app.db", "SELECT * FROM current_sessions \
                                          WHERE userid= ? AND session_magic = ?", (iuser, imagic))
    if len(row) > 0:
        return True
    return False


# Prevent user already logged in from new log in...
def handle_local_user_already_loggedin(iuser, imagic):
    """ Prevents a user already logged in from logging in again.
    """
    text = "<response>\n"
    text += build_response_refill('message',
                                  'User already logged in on this machine, \
                                  end current session to continue')
    user = iuser
    magic = imagic
    text += "</response>\n"
    return user, magic, text


# A user has supplied a username (parameters['usernameinput'][0])
# and password (parameters['passwordinput'][0]) check if these are
# valid and if so, create a suitable session record in the database
# with a random magic identifier that is returned.
# Return the username, magic identifier and the response action set.
def handle_login_request(iuser, imagic, parameters):
    """ Handle a user logging in.
    """
    if handle_validate(iuser, imagic):
        # the user is already logged in, return error message (via function)...
        # might make more sense to allow login in this case in but following
        # spec...
        user, magic, text = handle_local_user_already_loggedin(iuser, imagic)
        return [user, magic, text]
    #print(f'Parameters: {parameters}')
    text = "<response>\n"

    # Handle no username inputted...
    if ('usernameinput' not in parameters.keys()) or (
            'passwordinput' not in parameters.keys()):
        text += build_response_refill('message',
                                      'No username or password entered')
        user = iuser
        magic = imagic
        text += "</response>\n"
        return [user, magic, text]

    # Check password is valid
    username_input = parameters['usernameinput'][0]
    password_input = parameters['passwordinput'][0]
    password_input_hashed = hash_password(password_input)
    row = access_database_with_result_sec(
        "traffic_app.db",
        "SELECT * FROM user_pass \
                                          WHERE username= ? AND pass = ?",
        (username_input,
         password_input_hashed))
    session_magic = ''.join(
        random.choices(
            string.ascii_uppercase +
            string.digits,
            k=10))
    current_time = time.strftime(
        '%Y-%m-%d %H:%M:%S',
        time.localtime(
            time.time()))
    #print(f'row: {row}')
    # Check the user is not already logged in on another machine...
    if len(row) > 0:
        check_if_logged_in = access_database_with_result_sec(
            "traffic_app.db", "SELECT * FROM current_sessions WHERE userid= ?", (int(row[0][0]),))
        if len(check_if_logged_in) > 0:
            text += build_response_refill('message',
                                          'User already logged in on another machine')
            user = '!'
            magic = ''
            text += "</response>\n"
            return [user, magic, text]

    # User is valid (and not already logged in)
    if len(row) > 0:
        text += build_response_redirect('/page.html')
        user = row[0][0]
        magic = str(session_magic)

        # Add userid and current session magic to current sessions database...
        access_database_sec(
            "traffic_app.db", "INSERT INTO current_sessions(userid,session_magic) \
                            VALUES (?,?)", (user, magic))
        access_database_sec(
            "traffic_app.db", "INSERT INTO all_sessions(userid,session_magic,session_start) \
                            VALUES (?,?,?)", (user, magic, current_time))

    else:  # The user is not valid
        text += build_response_refill('message',
                                      'Invalid username or password')
        user = '!'
        magic = ''

    text += "</response>\n"
    return [user, magic, text]


# The user has requested a vehicle be added to the count
# parameters['locationinput'][0] the location to be recorded
# parameters['occupancyinput'][0] the occupant count to be recorded
# parameters['typeinput'][0] the type to be recorded
# Return the username, magic identifier (these can be empty  strings) and
# the response action set.
def handle_add_request(iuser, imagic, parameters):
    """ Handle a user adding a traffic observation to the database.
    """
    text = "<response>\n"
    vec_type = [
        'car',
        'bus',
        'bicycle',
        'motorbike',
        'van',
        'truck',
        'taxi',
        'other']
    occp_levels = ['1', '2', '3', '4']

    # Handle invalid cookies...
    if handle_validate(iuser, imagic) != True:
        text += build_response_refill('message',
                                      'Cannot add entry as user not logged in.')
        text += build_response_refill('total', '0')

    # Handle missing parameters...
    elif ('locationinput' not in parameters.keys()) or ('occupancyinput' not in parameters.keys()) or \
            ('typeinput' not in parameters.keys()):
        session_magic = imagic
        text += build_response_refill('message', 'Missing input')
        total_added = access_database_with_result_sec(
            "traffic_app.db", "SELECT count(session_magic) FROM traffic_records \
                                                      WHERE session_magic = ?", (session_magic,))

        total_undo = access_database_with_result_sec(
            "traffic_app.db", "SELECT count(session_magic) \
                                                     FROM records_to_undo WHERE session_magic = ?", (session_magic,))
        total_updated = int(total_added[0][0]) - int(total_undo[0][0])
        text += build_response_refill('total', f'{total_updated}')

    # Ensure parameters are valid...
    elif parameters['typeinput'][0] not in vec_type\
            or parameters['occupancyinput'][0] not in occp_levels \
            or check_location_input(parameters['locationinput'][0]) != True:
        session_magic = imagic
        text += build_response_refill('message', 'Invalid input(s)')
        total_added = access_database_with_result_sec(
            "traffic_app.db", "SELECT count(session_magic) FROM traffic_records \
                                                      WHERE session_magic = ?", (session_magic,))

        total_undo = access_database_with_result_sec(
            "traffic_app.db", "SELECT count(session_magic) \
                                                     FROM records_to_undo WHERE session_magic = ?", (session_magic,))
        total_updated = int(total_added[0][0]) - int(total_undo[0][0])
        text += build_response_refill('total', f'{total_updated}')

    # Parameters and cookies valid, proceed with addition...
    else:
        location = parameters['locationinput'][0]
        occupancy = parameters['occupancyinput'][0]
        vec_type = parameters['typeinput'][0]
        userid = iuser
        session_magic = imagic
        current_time = time.strftime(
            '%Y-%m-%d %H:%M:%S',
            time.localtime(
                time.time()))
        access_database_sec(
            "traffic_app.db",
            "INSERT INTO traffic_records(userid, session_magic, location, \
                            type, occupancy, entry_time) VALUES (?,?,?,?,?,?)",
            (userid,
             session_magic,
             location,
             vec_type,
             occupancy,
             current_time))
        total_added = access_database_with_result_sec(
            "traffic_app.db", "SELECT count(session_magic) FROM traffic_records \
                                                      WHERE session_magic = ?", (session_magic,))

        total_undo = access_database_with_result_sec(
            "traffic_app.db", "SELECT count(session_magic) \
                                                     FROM records_to_undo WHERE session_magic = ?", (session_magic,))

        total_updated = int(total_added[0][0]) - int(total_undo[0][0])

        text += build_response_refill('message', 'Entry added.')
        text += build_response_refill('total', f'{total_updated}')

    user = iuser
    magic = imagic
    text += "</response>\n"

    return [user, magic, text]


# The user has requested a vehicle be removed from the count
# This is intended to allow counters to correct errors.
# parameters['locationinput'][0] the location to be recorded
# parameters['occupancyinput'][0] the occupant count to be recorded
# parameters['typeinput'][0] the type to be recorded
# Return the username, magic identifier (these can be empty  strings) and
# the response action set.
def handle_undo_request(iuser, imagic, parameters):
    """ Handle a user removing a traffic observation to the database.
    """
    text = "<response>\n"
    if handle_validate(iuser, imagic) != True:
        # Invalid sessions redirect to login
        text += build_response_refill('message',
                                      'Cannot undo entry as user not logged in.')
        text += build_response_refill('total', '0')

    # Handle missing parameters...
    elif ('locationinput' not in parameters.keys()) or ('occupancyinput' not in parameters.keys()) or \
            ('typeinput' not in parameters.keys()):
        session_magic = imagic
        total_added = access_database_with_result_sec("traffic_app.db", "SELECT count(session_magic) \
                                                              FROM traffic_records WHERE session_magic = ?",
                                                      (session_magic,))
        total_undo = access_database_with_result_sec("traffic_app.db", "SELECT count(session_magic) \
                                                             FROM records_to_undo WHERE session_magic = ?",
                                                     (session_magic,))
        total_updated = int(total_added[0][0]) - int(total_undo[0][0])
        text += build_response_refill('message', 'Missing undo input')
        text += build_response_refill('total', f'{total_updated}')

    # a valid session with all parameters so process the addition of the
    # entry...
    else:
        location = parameters['locationinput'][0]
        occupancy = parameters['occupancyinput'][0]
        vec_type = parameters['typeinput'][0]
        userid = iuser
        session_magic = imagic

        # Check that the entry exists...
        check_record_exist = access_database_with_result_sec(
            "traffic_app.db",
            "SELECT entry_no FROM \
                                                             traffic_records WHERE userid = ? AND session_magic = ? \
                                                             AND location = ? AND type = ? AND occupancy = ?",
            (userid,
             session_magic,
             location,
             vec_type,
             occupancy))
        if len(check_record_exist) > 0:
            identical_records = [rec[0] for rec in check_record_exist]
            identical_records_undone = access_database_with_result_sec("traffic_app.db", "SELECT count(session_magic) \
                                                                       FROM records_to_undo WHERE userid = ? \
                                                                           AND session_magic = ? AND location = ? \
                                                                               AND type = ? AND occupancy = ?",
                                                                       (userid, session_magic, location, vec_type, occupancy))

            index = identical_records_undone[0][0]
            #print(f'identical records: {identical_records}')
            #print(f'index: {index}')
            if index >= len(identical_records):
                total_added = access_database_with_result_sec("traffic_app.db", "SELECT count(session_magic) \
                                                              FROM traffic_records WHERE session_magic = ?",
                                                              (session_magic,))
                total_undo = access_database_with_result_sec("traffic_app.db", "SELECT count(session_magic) \
                                                             FROM records_to_undo WHERE session_magic = ?",
                                                             (session_magic,))
                total_updated = int(total_added[0][0]) - int(total_undo[0][0])
                text += build_response_refill('message',
                                              'Entry does not exist')
                text += build_response_refill('total', f'{total_updated}')

            #print(f' Record exist check: {check_record_exists}')
            # If the entry exists, add it to the undo table and update total...
            elif len(check_record_exist) > 0:

                access_database_sec(
                    "traffic_app.db",
                    "INSERT INTO records_to_undo VALUES (?,?,?,?,?,?)",
                    (identical_records[index],
                     userid,
                     session_magic,
                     location,
                     vec_type,
                     occupancy))
                total_added = access_database_with_result_sec("traffic_app.db", "SELECT count(session_magic) \
                                                              FROM traffic_records WHERE session_magic = ?",
                                                              (session_magic,))
                total_undo = access_database_with_result_sec("traffic_app.db", "SELECT count(session_magic) \
                                                             FROM records_to_undo WHERE session_magic = ?",
                                                             (session_magic,))
                total_updated = int(total_added[0][0]) - int(total_undo[0][0])

                text += build_response_refill('message', 'Entry removed.')
                text += build_response_refill('total', f'{total_updated}')

        else:
            total_added = access_database_with_result_sec("traffic_app.db", "SELECT count(session_magic) \
                                                              FROM traffic_records WHERE session_magic = ?",
                                                          (session_magic,))
            total_undo = access_database_with_result_sec("traffic_app.db", "SELECT count(session_magic) \
                                                             FROM records_to_undo WHERE session_magic = ?",
                                                         (session_magic,))
            total_updated = int(total_added[0][0]) - int(total_undo[0][0])
            text += build_response_refill('message',
                                          'Entry does not exist')
            text += build_response_refill('total', f'{total_updated}')

    user = iuser
    magic = imagic
    text += "</response>\n"
    return [user, magic, text]


# This code handles the selection of the back button on the record form
# (page.html)
def handle_back_request(iuser, imagic, parameters):
    """ Handle a user pressing the back button (takes them to summary page).
    """
    text = "<response>\n"
    if handle_validate(iuser, imagic) != True:
        text += build_response_redirect('/index.html')
    else:
        text += build_response_redirect('/summary.html')
    text += "</response>\n"
    user = iuser
    magic = imagic
    return [user, magic, text]


# This code handles the selection of the logout button on the summary page (summary.html)
# You will need to ensure the end of the session is recorded in the database
# And that the session magic is revoked.
def handle_logout_request(iuser, imagic):
    """ Handles a user logging out.
    """
    text = "<response>\n"
    if handle_validate(iuser, imagic) != True:
        text += build_response_redirect('/index.html')
    else:
        # Add log out time to all_sessions table...
        current_time = time.strftime(
            '%Y-%m-%d %H:%M:%S',
            time.localtime(
                time.time()))
        access_database_sec("traffic_app.db", "UPDATE all_sessions SET session_end=? \
                            WHERE userid = ? AND session_magic = ?",
                            (current_time, iuser, imagic))

        # Clear current session magic from current sessions table...
        access_database_sec(
            "traffic_app.db", "DELETE FROM current_sessions WHERE userid = ? \
                            AND session_magic = ?", (iuser, imagic))
        text += build_response_refill('message', 'Logging out...')
        text += build_response_redirect('/index.html')
    user = '!'
    magic = ''

    text += "</response>\n"
    return [user, magic, text]


# This code handles a request for update to the session summary values.
# You will need to extract this information from the database.
def handle_summary_request(iuser, imagic):
    """ Handle generating the summary statistics page.
    """
    text = "<response>\n"
    if handle_validate(iuser, imagic) != True:
        text += build_response_redirect('/index.html')
    else:
        userid = iuser
        session_magic = imagic
        totals = {}
        vehicles = [
            'car',
            'taxi',
            'bus',
            'bicycle',
            'motorbike',
            'van',
            'truck',
            'other']

        for vehicle in vehicles:
            count_all = access_database_with_result_sec(
                "traffic_app.db",
                "SELECT count(entry_no) \
                                                        FROM traffic_records WHERE userid=? AND session_magic=? AND type=?",
                (userid,
                 session_magic,
                 vehicle))
            count_undo = access_database_with_result_sec(
                "traffic_app.db",
                "SELECT count(entry_no) \
                                                         FROM records_to_undo WHERE userid=? AND session_magic=? AND type=?",
                (userid,
                 session_magic,
                 vehicle))
            count = count_all[0][0] - count_undo[0][0]
            totals[vehicle] = count
        total = sum([v for k, v in totals.items()])

        text += build_response_refill('sum_car', f'{totals["car"]}')
        text += build_response_refill('sum_taxi', f'{totals["taxi"]}')
        text += build_response_refill('sum_bus', f'{totals["bus"]}')
        text += build_response_refill('sum_motorbike',
                                      f'{totals["motorbike"]}')
        text += build_response_refill('sum_bicycle', f'{totals["bicycle"]}')
        text += build_response_refill('sum_van', f'{totals["van"]}')
        text += build_response_refill('sum_truck', f'{totals["truck"]}')
        text += build_response_refill('sum_other', f'{totals["other"]}')
        text += build_response_refill('total', f'{total}')
        text += "</response>\n"
    user = iuser
    magic = imagic
    return [user, magic, text]


# HTTPRequestHandler class
class myHTTPServer_RequestHandler(BaseHTTPRequestHandler):

    # GET This function responds to GET requests to the web server.
    def do_GET(self):

        # The set_cookies function adds/updates two cookies returned with a webpage.
        # These identify the user who is logged in. The first parameter identifies the user
        # and the second should be used to verify the login session.
        def set_cookies(x, user, magic):
            ucookie = Cookie.SimpleCookie()
            ucookie['u_cookie'] = user
            x.send_header("Set-Cookie", ucookie.output(header='', sep=''))
            mcookie = Cookie.SimpleCookie()
            mcookie['m_cookie'] = magic
            x.send_header("Set-Cookie", mcookie.output(header='', sep=''))

        # The get_cookies function returns the values of the user and magic cookies if they exist
        # it returns empty strings if they do not.
        def get_cookies(source):
            rcookies = Cookie.SimpleCookie(source.headers.get('Cookie'))
            user = ''
            magic = ''
            for keyc, valuec in rcookies.items():
                if keyc == 'u_cookie':
                    user = valuec.value
                if keyc == 'm_cookie':
                    magic = valuec.value
            return [user, magic]

        # Fetch the cookies that arrived with the GET request
        # The identify the user session.
        user_magic = get_cookies(self)

        #print(f'Inital user magic: {user_magic}')

        # Parse the GET request to identify the file requested and the GET
        # parameters
        parsed_path = urllib.parse.urlparse(self.path)

        # Decided what to do based on the file requested.

        # Return a CSS (Cascading Style Sheet) file.
        # These tell the web client how the page should appear.
        if self.path.startswith('/css'):
            self.send_response(200)
            self.send_header('Content-type', 'text/css')
            self.end_headers()
            with open('.' + self.path, 'rb') as file:
                self.wfile.write(file.read())
            file.close()

        # Return a Javascript file.
        # These tell contain code that the web client can execute.
        if self.path.startswith('/js'):
            self.send_response(200)
            self.send_header('Content-type', 'text/js')
            self.end_headers()
            with open('.' + self.path, 'rb') as file:
                self.wfile.write(file.read())
            file.close()

        # A special case of '/' means return the index.html (homepage)
        # of a website
        elif parsed_path.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open('./index.html', 'rb') as file:
                self.wfile.write(file.read())
            file.close()

        # Return html pages.
        elif parsed_path.path.endswith('.html'):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            with open('.' + parsed_path.path, 'rb') as file:
                self.wfile.write(file.read())
            file.close()

        # The special file 'action' is not a real file, it indicates an action
        # we wish the server to execute.
        elif parsed_path.path == '/action':
            # respond that this is a valid page request
            self.send_response(200)
            # extract the parameters from the GET request.
            # These are passed to the handlers.
            parameters = urllib.parse.parse_qs(parsed_path.query)

            if 'command' in parameters:
                # check if one of the parameters was 'command'
                # If it is, identify which command and call the appropriate
                # handler function.
                if parameters['command'][0] == 'login':
                    [user, magic, text] = handle_login_request(
                        user_magic[0], user_magic[1], parameters)
                    # The result to a login attempt will be to set
                    # the cookies to identify the session.
                    set_cookies(self, user, magic)
                elif parameters['command'][0] == 'add':
                    [user, magic, text] = handle_add_request(
                        user_magic[0], user_magic[1], parameters)
                    # Check if we've been tasked with discarding the cookies.
                    if user == '!':
                        set_cookies(self, '', '')
                elif parameters['command'][0] == 'undo':
                    [user, magic, text] = handle_undo_request(
                        user_magic[0], user_magic[1], parameters)
                    # Check if we've been tasked with discarding the cookies.
                    if user == '!':
                        set_cookies(self, '', '')
                elif parameters['command'][0] == 'back':
                    [user, magic, text] = handle_back_request(
                        user_magic[0], user_magic[1], parameters)
                    # Check if we've been tasked with discarding the cookies.
                    if user == '!':
                        set_cookies(self, '', '')
                elif parameters['command'][0] == 'summary':
                    [user, magic, text] = handle_summary_request(
                        user_magic[0], user_magic[1])
                    # Check if we've been tasked with discarding the cookies.
                    if user == '!':
                        set_cookies(self, '', '')
                elif parameters['command'][0] == 'logout':
                    [user, magic, text] = handle_logout_request(
                        user_magic[0], user_magic[1])
                    # Check if we've been tasked with discarding the cookies.
                    if user == '!':
                        set_cookies(self, '', '')
                else:
                    # The command was not recognised, report that to the user.
                    text = "<response>\n"
                    text += build_response_refill('message',
                                                  'Internal Error: Command not recognised.')
                    text += "</response>\n"

            else:
                # There was no command present, report that to the user.
                text = "<response>\n"
                text += build_response_refill('message',
                                              'Internal Error: Command not found.')
                text += "</response>\n"
            self.send_header('Content-type', 'application/xml')
            self.end_headers()
            self.wfile.write(bytes(text, 'utf-8'))
        else:
            # A file that does n't fit one of the patterns above was requested.
            self.send_response(404)
            self.end_headers()

        #print('Updated magic:')
        #print(f'User: {user}, magic: {magic}')
        # print('Current sessions:')
        # print(
        #     access_database_with_result(
        #         "traffic_app.db",
        #         "SELECT * FROM current_sessions"))
        # print('All sessions:')
        # print(
        #     access_database_with_result(
        #         "traffic_app.db",
        #         "SELECT * FROM all_sessions"))
        # print('Traffic records:')
        # print(
        #     access_database_with_result(
        #         "traffic_app.db",
        #         "SELECT * FROM traffic_records"))
        # print('Records to undo:')
        # print(
        #     access_database_with_result(
        #         "traffic_app.db",
        #         "SELECT * FROM records_to_undo"))

        return


# This is the entry point function to this code.
def run():
    print('starting server...')
    # You can add any extra start up code here

    # Server settings
    # Choose port 8081 over port 80, which is normally used for a http server
    server_address = ('127.0.0.1', 8081)
    httpd = HTTPServer(server_address, myHTTPServer_RequestHandler)
    print('running server...')
    # This function will not return till the server is aborted.
    httpd.serve_forever()


run()
