import quandl
import math
import datetime
import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import os

style.use('ggplot')  # style for the graph

quandl.api_config.save_key(os.getenv('API_KEY_QUANDL'))
df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-9999, inplace=True)

forecast_out = int(math.ceil(0.1 * len(df)))
print(forecast_out)
df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]  # the features we will use to predict (we don't have a y value for these)
X = X[:-forecast_out]  # remove forecast_out amount of features out of list (not sure why we have this?)

df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)

#  starting prediction (make sure to check changes made above)

# prediction, can pass in one value or an array of values (X_lately)
forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

# visualising the data on a graph

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400  # seconds in a day
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    # .loc[next_date] refers to an index on df, (creates if no index or replaces if there is)
    # for the forecasted values, makes everything np.nan on all columns apart from the last index
    # which is the Forecast index and inserts the 'i' in each forecast spot
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

# doing the actual plotting
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
