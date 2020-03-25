# Project Title
C.A.R.E Web Application


## Setup
$ cd GitHub\fyp\Frontend\Code\CARE APP - Tanny (directory where wsgi.py resides in)
$ python3 setup.py install

## Run the Application
$ python wsgi.py
$ python server.py

Application is hosted on **http://127.0.0.1:5000/**

"Running on http://0.0.0.0:5000/" message on cmd or console informs you that your PC will be listening for incoming requests from all NICs/ bind to all interface.
(reference: https://stackoverflow.com/questions/46835568/this-site-can-t-be-reached-flask-python)  

## Documentation

### setup.py
Installation file


### wsgi.py
Serves the Flask application.


### __init__.py
Initializes python packages. Constructs the core application and registers them.


### server.py
Server socket running on localhost port 1236. Waits for request to come in/ connection from client.
Performs operation based on request and returns result to client socket.
Call machine learning model here.


### assets.py
Asset management. Integrate webassets into Flask app. 
Initiatlize the app by calling Environment instance and registering assets in the form of bundles.


### routes.py
Routes for core Flask app. Session started/ declared here. 
Client socket starts and initiates connection to server on form submit (POST request). Sends data from form to server socket. 
Results received from server socket is stored in session as cookies.


### layout.py
html layout for Dash application. Overrides the underlying HTML template of Dash application.


### dash_example.py
Python file that contains the Dash application. All python and dash code/ functions/ graphs to be written in this file.
Multi-Page app and url routing defined using app callbacks.


### static folder
Static assets such as CSS, js, bootstrap and images.
Referenced with url_for('static', filename='...')


### templates folder
Jinja2 templates rendered as HTML templates for Flask app. 
Reference with render_template('...')


