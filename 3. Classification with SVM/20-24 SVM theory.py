import numpy as np
from sklearn import preprocessing, model_selection, neighbors, svm
import pandas as pd

# starting with 13-14 K nearest neighbors.py

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -9999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = svm.SVC(kernel='linear')  # where we declare svm (SVC classifier), kernel fixes accuracy issue for now
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print(accuracy)


