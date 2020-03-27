"""Routes for core Flask app."""
from flask import Blueprint, render_template, request, jsonify, json, redirect, session, url_for
from flask import current_app as app
import pickle
import os


import socket
import sys
import json

# start session
SESSION_TYPE = 'memcache'
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'

main_bp = Blueprint('main_bp', __name__,
                    template_folder='templates',
                    static_folder='static'
           )


@main_bp.route('/')
def home():
    """Landing page."""
    return render_template('landing.html',
                           title='C.A.R.E',
                        )

@app.route('/index2.html')
def index():
    return render_template('index2.html')


#request form data + client socket intiates connection
@app.route('/submit', methods=['POST'])
def submit():
    # receive data from form POST request. request.form returns json object
    req = request.form
    data = json.dumps(req) 

    # Create a TCP/ IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    BUFF_SIZE = 4096
    full_message =b''

    # Connect socket to the port where the server is listening
    server_address = ('localhost', 1236)
    print('connecting to {} port {}'.format(*server_address))

    sock.connect(server_address)

    try:
        # send form data
        message = bytes(data, encoding='utf-8')
        print('sending {!r}'.format(message))
        sock.sendall(message)

        # looks for the response
        amount_received = 0
        amount_expected = len(message)

        while True:
            received = sock.recv(BUFF_SIZE)
            full_message += received

            #amount_received += len(received)
            if len(part) < BUFF_SIZE: 
                break            
        
    finally:
        print('received {!r}'.format(received))
        print('closing socket')
        sock.close()
        session['received'] = full_message
        print(session)
        return redirect('/survival/')


# dynamic timestamp variable for static css/ js file to resolve browser cache issue (css doesn't load on app change.)
@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                 endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)


