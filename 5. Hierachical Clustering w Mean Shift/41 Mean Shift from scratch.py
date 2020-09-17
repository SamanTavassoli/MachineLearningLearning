import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11],
              [8, 2],
              [10, 2],
              [9, 3]])

colors = ["g", "r", "c", "b", "k", "o"]


class Mean_Shift:
    def __init__(self, radius=4):  # important that radius must be set correctly (next tuto)
        self.radius = radius

    def fit(self, data):
        centroids = {}

        for i in range(len(data)):
            # id of i, value is datapoint
            centroids[i] = data[i]

        while True:
            new_centroids = []

            for i in centroids:
                # contains all feature_sets within the radius of the centroid
                in_bandwidth = []
                centroid = centroids[i]
                for feature_set in data:
                    # np.norm just to get the magnitude of the difference vector
                    if np.linalg.norm(feature_set - centroid) < self.radius:
                        in_bandwidth.append(feature_set)

                # getting the mean vector from all vectors within radius
                new_centroid = np.average(in_bandwidth, axis=0)
                new_centroids.append(tuple(new_centroid))  # tuple to be able to use set() below

            # sorted list of unique new centroids
            # sorted because we want to compare previous centroids later and it's easier if they are in the same order
            uniques = sorted(list(set(new_centroids)))

            prev_centroids = dict(centroids)  # to check if we have found the final centroids

            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])

            # optimized if the centroids are not changing
            optimized = True
            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False

                if not optimized:  # if one has moved, no need to continue
                    break
            if optimized:  # but here we break while loop cause we have obtained the final centroids
                break

        self.centroids = centroids  # so we can get the centroids after fitting

    def predict(self, data):
        pass


clf = Mean_Shift()
clf.fit(X)

centroids = clf.centroids

plt.scatter(X[:, 0], X[:, 1], s=150)

for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='*', s=150)

plt.show()
