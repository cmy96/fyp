"""Routes for core Flask app."""
from flask import Blueprint, render_template, request, jsonify, json, redirect, session, url_for
from flask import current_app as app
import pickle


import socket
import sys
import json

SESSION_TYPE = 'memcache'
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'

main_bp = Blueprint('main_bp', __name__,
                    template_folder='templates',
                    static_folder='static')


@main_bp.route('/')
def home():
    """Landing page."""
    return render_template('landing.html',
                           title='C.A.R.E',
                           template='home-template',
                           body="")

@app.route('/index2.html')
def index():
    return render_template('index2.html')


@app.route('/submit', methods=['POST'])
def submit():
    req = request.form
    data = json.dumps(req) 
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    BUFF_SIZE = 4096
    full_message =b''

    server_address = ('localhost', 1236)
    print('connecting to {} port {}'.format(*server_address))

    sock.connect(server_address)

    try:
        message = bytes(data, encoding='utf-8')
        print('sending {!r}'.format(message))
        sock.sendall(message)

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
        return redirect('/results/')




