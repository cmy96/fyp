# Project Title
C.A.R.E Web Application

Requirement: Download Anaconda on the computer

Installation: Using Anaconda prompt run all the install commands below:
conda install -c anaconda flask
conda install -c conda-forge dash
conda install -c conda-forge dash-bootstrap-components
pip install survive
conda install -c sebp scikit-survival
conda install -c conda-forge tensorflow
c drive > users > users> anaconda3 > envs> fyp > lib > site-packages > font (add font5 awesome.otf file)
conda install -c conda-forge flask-assets


## Setup: open 2 anaconda command prompts and run the line below for both command prompts
$ cd GitHub\fyp\Frontend\Code\CARE APP - Tanny (the directory where wsgi.py resides in) 

## Run the Application
$ python wsgi.py
$ python server.py

When the Application is loaded, you will see the message below:

"Running on http://0.0.0.0:5000/" message on cmd or console informs you that your PC will be listening for incoming requests from all NICs/ bind to all interface.
(reference: https://stackoverflow.com/questions/46835568/this-site-can-t-be-reached-flask-python)  

Open a browser, the application will be hosted on http://127.0.0.1:5000/


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

#### To run prediction models
1) conda install -c sebp scikit-survival
2) conda install tensorflow=2.0 in anaconda env
3) Open an empty folder - middleWomen (case-sensitive) in the same level of MAINAPP
4) Ensure you have the necessary data files in /data/ (kapalan_meier_by_group.csv)
5) make sure you have the most updated files - server.py & somefunction.py & dash_example.py
6) make sure you have the 4 folders OHE, Model_folder, ann, Layered_folder in C:/SMU_v2/
7) Check all references to file directories (if the files exist)


### static folder
Static assets such as CSS, js, bootstrap and images.
Referenced with url_for('static', filename='...')


### templates folder
Jinja2 templates rendered as HTML templates for Flask app. 
Reference with render_template('...')