import math

import pandas as pd
import quandl
import numpy as np
from sklearn import preprocessing, model_selection
from sklearn.linear_model import LinearRegression
import os

quandl.api_config.save_key(os.getenv('API_KEY_QUANDL'))
df = quandl.get_table('RSM/MSB')  # will try to predict the number of finished products based on chosen features

df = df[['employeecars', 'containers', 'trucks', 'tippers', 'finishedproducts']]

# not using type but this is how we would cast it to int
# df['type'] = df['type'].map(lambda x: x.lstrip('MSS_')).convert_dtypes(convert_integer=True)

print(df)
forecast_col = 'finishedproducts'
df.fillna(-9999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))
df['label'] = df[forecast_col].shift(-forecast_out) # added 14 Apr 2021 - not much change really, kind of bad data for this
df.dropna(inplace=True)

X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

print(accuracy)
# accuracy is terrible cause we need more data but whatever
