import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('ggplot')

X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8],
              [8, 8],
              [1, 0.6],
              [9, 11]])

plt.scatter(X[:, 0], X[:, 1], s=150)

colors = ["g", "r", "c", "b", "k", "o"]


def plot():
    for centroid in clf.centroids:
        plt.scatter(clf.centroids[centroid][0], clf.centroids[centroid][1], marker="o", color="k", linewidths=5)

    for classification in clf.classifications:
        color = colors[classification]  # color based on classification
        for feature_set in clf.classifications[classification]:
            plt.scatter(feature_set[0], feature_set[1], marker="x", color=color, s=150, linewidths=5)

    plt.show()

class K_Means:
    # we're using the same default values for the params as sklearn
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):

        self.centroids = {}

        for i in range(self.k):
            self.centroids[i] = data[i]  # first two centroids are the first two points of the data (doesn't rly matter)

        for i in range(self.max_iter):
            # which centroids does each feature_set belong to {centroid : feature_set}
            # redefine classifications everytime the centroid changes (each iteration)
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for feature_set in data:
                # average distance for each feature_set to each centroid (k centroids)
                distances = [np.linalg.norm(feature_set - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))  # which centroid is it closest to?
                # tag with classification and put in dictionary
                self.classifications[classification].append(feature_set)

            prev_centroids = dict(self.centroids)  # to compare previous centroid to new centroid using tolerance value

            # this is where the centroids are correcting themselves and moving around
            # each centroid moves according the new center of the current feature_sets that are classified as
            # belonging to that centroid
            for classification in self.classifications:
                plot()  # can see what is happening at each step by plotting here
                # axis=0 used below to obtain the average for each coordinate
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            # if the difference of the centroids is less than tolerance we can proceed
            optimized = True
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                # here np.sum sums the different coordinates for each dimension
                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tol:
                    optimized = False

            if optimized:
                break

    # predict a point based on which centroid it is closest to
    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


clf = K_Means()
clf.fit(X)

unknowns = np.array([[1, 3],
                     [8, 9],
                     [0, 3],
                     [5, 4],
                     [6, 4]])

for unknown in unknowns:
    classification = clf.predict(unknown)
    plt.scatter(unknown[0], unknown[1], marker="*", color=colors[classification], s=150, linewidths=5)

plot()

