import numpy as np
import matplotlib as plt
from matplotlib import style
from sklearn import preprocessing
import pandas as pd
import xdrlib

style.use('ggplot')


class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):

        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for feature_set in data:
                distances = [np.linalg.norm(feature_set - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(feature_set)

            previous_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = previous_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tol:
                    optimized = False

            if optimized:
                break

    def predict(self, data_point):
        distances = [np.linalg.norm(data_point - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


def handle_non_num_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_values = {}

        def convert_to_int(val):
            return text_digit_values[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique_element in unique_elements:
                text_digit_values[unique_element] = x
                x += 1

            df[column] = list(map(convert_to_int, df[column]))

    return df


df = pd.read_excel('titanic.xls')
df.drop(['body', 'name'], 1, inplace=True)
df.fillna(0, inplace=True)
df = handle_non_num_data(df)
X = np.array(df.drop(['survived'], 1)).astype(float)
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = K_Means()
clf.fit(X)

correct = 0

for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction == y[i]:
        correct += 1

print(correct / len(X))





