from math import sqrt
import numpy as np
from collections import Counter
import warnings
import pandas as pd
import random

# sklearn only for looking at the difference with our own implementation
from sklearn import preprocessing, model_selection, neighbors


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less that total voting groups! idiot!')

    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    vote_result = Counter(votes).most_common(1)[0][0]

    # adding confidence -> how many of the most common (1)[0][1] divided by k
    confidence = Counter(votes).most_common(1)[0][1] / k

    return vote_result, confidence


df = pd.read_csv("breast-cancer-wisconsin.data")
df.replace('?', -9999, inplace=True)
df.drop('id', 1, inplace=True)

number_of_reps = 100
accuracies = []
for i in range(number_of_reps):
    full_data = df.astype(float).values.tolist()  # conversion to list of list of floats (might not be necessary)
    random.shuffle(full_data)

    test_size = 0.2
    # the indices of the dictionary are the labels while the corresponding lists contain the features
    train_set = {2: [], 4: []}
    test_set = {2: [], 4: []}
    train_data = full_data[:-int(test_size * len(full_data))]  # everything except last test_size*100 % of data
    test_data = full_data[-int(test_size * len(full_data)):]  # last test_size*100 % of data

    for i in train_data:
        # choose the group in train_set to which you will add data (minus last column)
        # based on the last column of the data
        train_set[i[-1]].append(i[:-1])
    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    correct = 0
    total = 0

    for group in test_set:
        for data in test_set[group]:
            # for each test point, takes k=5 nearest using the training set
            vote, confidence = k_nearest_neighbors(train_set, data, k=5)  # default of 5 for k
            if group == vote:  # checking vote against label
                correct += 1
            # can also do something with the confidence if we want
            total += 1
    print('Accuracy: ', correct / total)
    accuracies.append(correct / total)

print('Accuracy over ' + str(len(accuracies)) + ' tests: ' + str(sum(accuracies)/len(accuracies)))


# ----- looking at the sklearn results for comparison
# we wait a lot longer on our own version

accuracies = []
for i in range(number_of_reps):

    X = np.array(df.drop(['class'], 1))
    y = np.array(df['class'])
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

    clf = neighbors.KNeighborsClassifier(n_jobs=-1)
    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)
    accuracies.append(accuracy)

print('Sklearn accuracy: ' + str(sum(accuracies)/len(accuracies)))




