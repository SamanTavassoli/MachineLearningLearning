import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from sklearn.datasets import make_blobs

style.use('ggplot')

X, y = make_blobs(n_samples=50, centers=4, n_features=2)

# X = np.array([[1, 2],
#               [1.5, 1.8],
#               [5, 8],
#               [8, 8],
#               [1, 0.6],
#               [9, 11],
#               [8, 2],
#               [10, 2],
#               [9, 3]])

colors = 100*["g", "r", "c", "b", "k"]


class Mean_Shift:
    # new change, radius set to none
    def __init__(self, radius=None, radius_norm_step=100):
        self.radius = radius
        self.radius_norm_step = radius_norm_step

    def fit(self, data):

        if self.radius is None:
            # just the one centroid in the center of all data
            all_data_centroid = np.average(data, axis=0)
            all_data_norm = np.linalg.norm(all_data_centroid)
            # radius as a ratio of the overall centroid and the number of steps
            self.radius = all_data_norm / self.radius_norm_step


        centroids = {}

        for i in range(len(data)):
            centroids[i] = data[i]

        # [::-1] reverses the list (now 99, 98, 97...)
        weights = [i for i in range(self.radius_norm_step)][::-1]

        while True:
            new_centroids = []

            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]
                for feature_set in data:
                    distance = np.linalg.norm(feature_set - centroid)
                    if distance == 0:
                        distance = 0.00000001  # for first iteration
                    # weight index -> how many radii in this distance?
                    # the larger weight index, the more it will affect centroid
                    weight_index = int(distance / self.radius)
                    # if weight is more than 100 steps away, set it to 100 (weight index would be 0)
                    # we want the weight to be somewhere in the weights for most points (closer has higher weight index)
                    if weight_index > self.radius_norm_step - 1:
                        weight_index = self.radius_norm_step - 1
                    # weighing feature set
                    # penalising for weights that are farther with **2 (but reduce efficiency doing it this way)
                    # each centroids we are checking is going to be much more affected by feature sets closer to it
                    to_add = (weights[weight_index]**2)*[feature_set]
                    in_bandwidth += to_add

                new_centroid = np.average(in_bandwidth, axis=0)
                new_centroids.append(tuple(new_centroid))

            uniques = sorted(list(set(new_centroids)))

            to_pop = []

            # checking if some centroids are close enough to each other that they can be removed
            # comparing each point of uniques to each other point in uniques
            for i in uniques:
                for ii in uniques:
                    if i == ii:
                        pass  # can't get any of these cause we used set() above (can't be EXACTLY the same)
                    # removing centroids that are too close
                    elif np.linalg.norm(np.array(i) - np.array(ii)) <= self.radius:
                        to_pop.append(ii)
                        break

            for i in to_pop:  # can't remove within a loop so we need new loop
                # we need to try here because I think there can be two instances of i in to remove due
                # to the method of iteration
                try:
                    uniques.remove(i)
                except:
                    pass

            prev_centroids = dict(centroids)

            centroids = {}
            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])

            optimized = True
            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False

                if not optimized:
                    break
            if optimized:
                break

        self.centroids = centroids

        self.classifications = {}  # {0:[feature sets], 1:[feature sets], 2...}

        for i in range(len(self.centroids)):
            self.classifications[i] = []

        for feature_set in data:
            distances = [np.linalg.norm(feature_set - self.centroids[centroid]) for centroid in self.centroids]
            classification = distances.index(min(distances))  # what is the closest centroid -> classified as that
            self.classifications[classification].append(feature_set)

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid] for centroid in self.centroids)]
        classification = distances.index(min(distances))
        return classification


clf = Mean_Shift()
clf.fit(X)

centroids = clf.centroids

for classification in clf.classifications:
    color = colors[classification]
    for feature_set in clf.classifications[classification]:
        plt.scatter(feature_set[0], feature_set[1], marker='x', color=color, s=150, linewidths=5)

for c in centroids:
    plt.scatter(centroids[c][0], centroids[c][1], color='k', marker='*', s=150)

plt.show()
