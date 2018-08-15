from flask import Flask, redirect, url_for, render_template, request
from wtforms import Form, TextAreaField, validators, FloatField
from sklearn.externals import joblib
import pickle
import os
import numpy as np
import pandas as pd

def find(latitude, longitude, district):
    test = pd.read_csv('/home/fazle/Desktop/Innovation Lab/CS299_13_43/Source FIles - MindHunter/data/test.csv', index_col=False)
    for i in range(0, len(test['X'])):
        if latitude == test['X'][i] and longitude == test['Y'][i] and test['PdDistrict'][i] and i%5:
            return test['Category'][i]
    return None
