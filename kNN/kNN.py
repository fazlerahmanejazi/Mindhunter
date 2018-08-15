# Loading libraries
import pandas as pd
from sklearn import preprocessing, cross_validation
from sklearn.neighbors import KNeighborsClassifier

# Specify path for files
path = "/home/fazle/Desktop/Innovation Lab/CS299_13_43/Source FIles - MindHunter/data/"
train_file = 'train.csv'
test_file = 'test.csv'
result_file = 'res.csv'

# Import training and test data.
train = pd.read_csv(path+test_file, parse_dates=['Dates'])
test = pd.read_csv(path+test_file, parse_dates=['Dates'])

def feature_engineering(data):
    data['Hour'] = data['Dates'].dt.hour
    day_of_week_encoded = preprocessing.LabelEncoder()
    day_of_week_encoded.fit(data['DayOfWeek'])
    data['DayOfWeekEncoded']=day_of_week_encoded.transform(data['DayOfWeek'])
    pd_district_encoded = preprocessing.LabelEncoder()
    pd_district_encoded.fit(train['PdDistrict'])
    train['PdDistrictEncoded'] = pd_district_encoded.transform(train['PdDistrict'])
    return data

def predict(data):
    scaler = preprocessing.StandardScaler().fit(train[['DayOfWeekEncoded', 'PdDistrictEncoded', 'X', 'Y', 'Hour']])
    knn = KNeighborsClassifier(n_neighbors = 5)
    knn.fit(scaler.transform(train[['DayOfWeekEncoded', 'PdDistrictEncoded', 'X', 'Y', 'Hour']]), train['Category'])
    result = knn.predict(scaler.transform(train[['DayOfWeekEncoded', 'PdDistrictEncoded', 'X', 'Y', 'Hour']]))
    return result

# Predicting and printing the result
train = feature_engineering(train)
test = feature_engineering(test)
test['Prediction'] = predict(test)
result = pd.DataFrame(test['Prediction'])
print('Accuracy :', sum(test['Category'] == test['Prediction']) / len(test['Dates']))
result.to_csv(result_file, index=False)
