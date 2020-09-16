import math
import quandl
import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import os

# ---- getting data, setting up data frame

# fetching data
quandl.api_config.save_key(os.getenv('API_KEY_QUANDL'))
df = quandl.get('WIKI/GOOGL')

# grabbing the features we want to use
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

# adding some more meaningful features
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

# ---- choosing features and labels we want to use to train

# keeping the features that we want to use
# NOTE The features used apart from Close have very little to do with price and are kind of useless
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

# starting to work on the column which is what we want to predict
forecast_col = 'Adj. Close'
df.fillna(-9999, inplace=True)  # Data that does not exist will be treated as an outlier, we'll see why -9999 is used
# later based on the algorithms

forecast_out = int(math.ceil(0.01 * len(df)))  # How many days forward we are looking
# ex 1002 objects, length -> 1002, *0.01 -> 10.02, math.ceil -> 11 (data from 11 days before predicts the current day)

# creating our label -> each label for a given day will be the adjusted close for example 10 days in the future
# "features are what may cause the adjusted close price in (example 10 days) to change", this is why the label is used
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)  # I think this is for when values are shifted and there is no label for the latest data

# ---- training and testing

# features are X, labels are y
X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])
X = preprocessing.scale(X)  # not sure what this is yet

# defining what portions of our data we want to use for training and which for testing
# 20% of data is used for test
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

# classifier uses the created training data to create a model
clf = LinearRegression(n_jobs=-1)  # this is where you pick your algorithm
# you can use the n_jobs option to speed up the training process by using more threads
clf.fit(X_train, y_train)
# score tests how well the model works against the testing data
accuracy = clf.score(X_test, y_test)

print(accuracy)




