"""Routes for core Flask app."""
from flask import Blueprint, render_template, request, jsonify, json
from flask import current_app as app
import pickle

import pandas as pd
import numpy as numpy
from sklearn.externals import joblib


#lr=joblib.load("C:\wamp64\www\fyp2\application\dash_application\model.pkl")

model = pickle.load(open('application/model.pkl','rb'))


main_bp = Blueprint('main_bp', __name__,
                    template_folder='templates',
                    static_folder='static')


@main_bp.route('/')
def home():
    """Landing page."""
    return render_template('index2.html',
                           title='C.A.R.E',
                           template='home-template',
                           body="")



@app.route('/submit', methods=['POST', 'GET'])
def submit():
    data = jsonify(request.form)

    return data

# def predict():

#     form_data = jsonify(request.form)

#     # get data
#     data = request.get_json(force=True)


#     # convert data into dataframe
#     data.update((x, [y]) for x, y in data.items())
#     data_df = pd.DataFrame.from_dict(data)

#     # predictions
#     result = model.predict(data_df)

#     # send back to browser
#     output = {'results': int(result[0])}

#     # return data
#     return jsonify(results=output)

