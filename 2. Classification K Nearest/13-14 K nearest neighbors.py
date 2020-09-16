import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd

# data set obtained from archive.ics.uci.edu/ml/datasets and column headers were added

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -9999, inplace=True)
df.drop(['id'], 1, inplace=True)  # K nearest neighbors doesn't handle outliers well

X = np.array(df.drop(['class'], 1))  # features
y = np.array(df['class'])  # labels

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()  # setting algorithm to K neighbors
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)

example_measures = np.array([[4, 2, 1, 1, 1, 2, 3, 2, 1], [4, 2, 1, 2, 2, 2, 3, 2, 1]])

prediction = clf.predict(example_measures)
print(prediction)

