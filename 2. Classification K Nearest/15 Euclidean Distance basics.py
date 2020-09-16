from math import sqrt

plot1 = [1, 3]
plot2 = [2, 5]

# from euclidean distance equation, for loop takes care of the n dimensions
euclidean_distance = sqrt(sum([(plot1[n] - plot2[n])**2 for n in range(len(plot1))]))

print(euclidean_distance)

