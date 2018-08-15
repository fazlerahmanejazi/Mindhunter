from flask import Flask, redirect, url_for, render_template, request
from wtforms import Form, TextAreaField, validators, FloatField
from sklearn.externals import joblib
import pickle
import os
import numpy as np
import pandas as pd
from util import find

app = Flask(__name__, static_url_path='/static')

######## Preparing the Classifier
cur_dir = os.path.dirname(__file__)
clf = joblib.load('rdf_model.pkl')
col_names = joblib.load('col_names.pkl')


######## Global variables
month_list = ['Jan', 'Feb', 'March', 'April', 'May', 'June', 'July', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
neighborhood_list = ['BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION', 'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']
hour_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]


######## Flask
class ReviewForm(Form):
    Latitude = FloatField(label='latitude(i.e., 37.733)', default='37.733', validators= [validators.InputRequired()])
    Longitude = FloatField(label='longitude(i.e., -122.394)', default='-122.394', validators= [validators.InputRequired()])


def classify(document):
    pkl_file = open('crime_categories.pkl', 'rb')
    crime = pickle.load(pkl_file)
    feat_array = np.zeros(len(col_names))
    for f in document:
        arr_index = np.where(col_names == f)
        feat_array[arr_index[0]] = 1
    feat_array[-3] = document[2]
    feat_array[-2] = document[3]
    feat_array[-1] = document[4]
    h = feat_array.reshape(len(feat_array), 1)
    X = pd.DataFrame(h.T, columns=col_names)
    y = clf.predict(X)
    result = clf.predict_proba(X)
    proba = np.max(result)
    return crime[y[0]], proba

@app.route('/')
def index():
    form = ReviewForm(request.form)
    return render_template('index.html', form=form, month_list=month_list, hour_list=hour_list, neighborhood_list=neighborhood_list)


@app.route("/test" , methods=['GET', 'POST'])
def test():
    form = ReviewForm(request.form)
    latitude = form.Latitude.data
    longitude = form.Longitude.data
    month = request.form.get('month')
    neighborhood = request.form.get('neighborhood')
    hour = request.form.get('hour')
    data = [month, neighborhood, longitude, latitude, int(hour)]
    return render_template('test.html', data=data)


@app.route('/results', methods=['POST'])
def results():
    form = ReviewForm(request.form)
    if request.method == 'POST':
        latitude = form.Latitude.data
        longitude = form.Longitude.data
        month = request.form.get('month')
        neighborhood = request.form.get('neighborhood')
        hour = request.form.get('hour')
        data = [month, neighborhood, longitude, latitude, int(hour)]
        x = find(latitude, longitude, neighborhood)
        if x :
            return render_template('results.html', data=data, prediction=x, probability=round(100, 2))
        y, proba = classify(data)
        return render_template('results.html', data=data, prediction=y, probability=round(proba*100, 2))
    return render_template('index.html', form=form, month_list=month_list, hour_list=hour_list, neighborhood_list=neighborhood_list)


if __name__ == '__main__':
    app.run(debug=True)
