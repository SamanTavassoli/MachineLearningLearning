from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings
style.use('fivethirtyeight')

# two groups (k, r) of 3 2-dimensional points
dataset = {'k': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}
new_features = [5, 7]


# classified data, point who's class is to be predicted, k points to check when predicting class
def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less that total voting groups! idiot!')

    distances = []
    for group in data:
        for features in data[group]:
            # using numpy to make calculation faster:
            # euclidean_distance = np.sqrt(np.sum((np.array(features)-np.array(predict))**2))
            # and even faster:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])
            # distances will look something like:
            # [[distance_1, group_1], [distance_2, group_1], [distance_1, group_2], [distance_1, group_2]]

    # sorting distances and keeping k number of distances cause that's all we need
    # the votes array ends up being a list of groups (we don't need the actual distances)
    votes = [i[1] for i in sorted(distances)[:k]]
    # (1) only want list of most common group
    # [0] to get first most common group
    # [0][0] to get vote, [0][1] to get number of counts towards that vote
    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result


result = k_nearest_neighbors(dataset, new_features, k=3)
print(result)

# for each group i, plot the points in that group in the group's color
[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0], new_features[1], s=100, color=result)  # coloring makes the outcome of algorithm obvious
plt.show()
