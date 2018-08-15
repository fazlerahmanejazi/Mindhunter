# Loading libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier

# Specify path for files
path = "/home/fazle/Desktop/Innovation Lab/CS299_13_43/Source FIles - MindHunter/data/"
train_file = 'train.csv'
test_file = 'test.csv'
result_file = 'res.csv'

# Import training and test data.
train = pd.read_csv(path+train_file, parse_dates=['Dates'], index_col=False)
test = pd.read_csv(path+test_file, parse_dates=['Dates'], index_col=False)

# Global variables
train.info()
categories = {c:i for i, c in enumerate(train['Category'].unique())}
categories_rev = {i:c for i, c in enumerate(train['Category'].unique())}
districts = {c:i for i, c in enumerate(train['PdDistrict'].unique())}
districts_rev = {i:c for i, c in enumerate(train['PdDistrict'].unique())}
weekdays = {'Monday':0., 'Tuesday':1., 'Wednesday':2., 'Thursday': 3., 'Friday':4., 'Saturday':5., 'Sunday':6.}

def define_address(addr):
    #Checking if address is intersecting or disjoint
    addr_type = 0.
    if '/' in addr and 'of' not in addr:
        addr_type = 1.
    else:
        add_type = 0.
    return addr_type

def getHourZn(hour):
    if(hour >= 2 and hour < 8): return 1 ;
    if(hour >= 8 and hour < 12): return 2 ;
    if(hour >= 12 and hour < 14): return 3 ;
    if(hour >= 14 and hour < 18): return 4 ;
    if(hour >= 18 and hour < 22): return 5 ;
    if(hour < 2 or hour >= 22): return 6 ;

def feature_engineering(data):
    data['X'] = preprocessing.scale(list(map(lambda x: x+122.4194, data.X)))
    data['Y'] = preprocessing.scale(list(map(lambda x: x-37.7749, data.Y)))
    data['Day'] = data['Dates'].dt.day
    data['Month'] = data['Dates'].dt.month
    data['Year'] = data['Dates'].dt.year-2003
    data['Hour'] = data['Dates'].dt.hour
    data['Minute'] = data['Dates'].dt.minute
    data['Day_Num'] = [float(weekdays[w]) for w in data.DayOfWeek]
    data['HourZn'] = preprocessing.scale(list(map(getHourZn, data['Dates'].dt.hour)))
    data['District_Num'] = [float(districts[t]) for t in data.PdDistrict]
    data['Address_Type'] = list(map(define_address, data.Address))
    return data

def predict(train, test):
    location_features = ['X', 'Y', 'District_Num', 'Address_Type']
    time_features = ['Minute', 'Hour', 'HourZn']
    date_features = ['Year','Month', 'Day', 'Day_Num']
    all_features = location_features + time_features + date_features
    train = feature_engineering(train)
    train['Category_Num'] = [float(categories[t]) for t in train.Category]
    test = feature_engineering(test)
    clf = RandomForestClassifier(max_features="log2", max_depth=100, n_estimators=30, min_samples_split=5, oob_score=True)
    clf = clf.fit(train[all_features], train['Category_Num'])
    predictor = clf.predict_proba(test[all_features])
    return predictor

def result(y, train):
    result = pd.DataFrame({categories_rev[p] : [y[i][p] for i in range(len(y))] for p in range(len(y[0]))})
    result['Id'] = [i for i in range(len(result))]
    result = result[['Id'] + sorted(train['Category'].unique())]
    result.to_csv(result_file, index=False)
    print("Done")

# Predicting and printing the result
predictor = predict(train, test)
result(predictor, train)
