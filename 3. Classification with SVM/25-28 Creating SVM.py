import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')


class Support_Vector_Machine:
    # visualisation for graphing
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    # fit is just training
    def fit(self, data):
        self.data = data
        # { ||W||: [w, b] }
        opt_dict = {}

        # need to apply all 4 transformations
        transforms = [[1, 1],
                      [-1, 1],
                      [-1, -1],
                      [1, -1]]

        all_data = []
        for yi in self.data:  # classes (-1, 1)
            for featureset in self.data[yi]:  # grabbing list from dictionary key
                for feature in featureset:  # grabbing points from list
                    all_data.append(feature)

        # all of the above just to get the max and the min of the data
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)
        all_data = None  # removing from memory

        # support vectors yi(xi.w+b) = 1
        # as close to 1 as possible depending on precision required (ex. 1.01)

        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      # point of expense:
                      self.max_feature_value * 0.001]

        # extremely expensive (it's not as valuable to get b as precise as w)
        b_range_multiple = 5
        # we don't need to take as small of steps as we do w (5 times greater steps
        # you wouldn't need this if you were using the same stepping with b as with w (which we aren't
        # because our data isn't large enough for it to matter)
        b_multiple = 5
        # major corner we are cutting
        optimum_value_multiplier = 10
        latest_optimum = self.max_feature_value * optimum_value_multiplier

        for step in step_sizes:
            # start is the corner we are cutting, starting coordinate?
            w = np.array([latest_optimum, latest_optimum])

            # we can do this because it's a convex problem (we know when it's optimized (past minimum of curve))
            optimized = False

            while not optimized:
                for b in np.arange(-1 * self.max_feature_value * b_range_multiple,
                                   self.max_feature_value * b_range_multiple,
                                   step * b_multiple):
                    for transformation in transforms:
                        w_t = w * transformation
                        found_option = True
                        # weakest link in the SVM fundamentally
                        # SMO attempts to fix this a bit
                        # constraint function yi(xi.w+b) >= 1
                        #
                        # #### a break can be added after found_option = False
                        for i in self.data:  # i is class (-1 or 1)
                            for xi in self.data[i]:
                                yi = i
                                if not yi*(np.dot(w_t, xi) + b) >= 1:
                                    # if even one sample does not fit constraint, throw out transformation
                                    found_option = False

                        if found_option:
                            # norm gives you magnitude of vector
                            # add the magnitude you obtained for w_t and b checked
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]

                # w[0] < 0 when we have gone past minimum so we stop
                # otherwise we take a step
                if w[0] < 0:
                    optimized = True
                    print('Optimized a step...')
                else:
                    # w = [5, 5]
                    # step = 1
                    # w - step  = [4, 4]
                    w = w - step

            # Setting up for smaller steps: finding the smallest norm, choosing best w and b, choosing latest_optimum

            # sorting by magnitudes
            norms = sorted([n for n in opt_dict])
            # choose smallest norm (bottom of the curve if you think of the curve)
            opt_choice = opt_dict[norms[0]]
            # ||w||: [w, b]
            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step  # * optimum_value_multiplier -> not sure why this is here

    def predict(self, features):
        # sing of x.w+b
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)

        # visualisation
        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=100, marker='*', c=self.colors[classification])

        return classification

    # no bearing on SVM, just to be able to see
    def visualise(self):
        [[self.ax.scatter(x[0], x[1], s=100, color=self.colors[i]) for x in data_dict[i]] for i in data_dict]

        # hyperplane = x.w+b
        # v = x.w+b
        # positive_support_vector -> v = 1
        # negative_support_vector -> v = -1
        # decision_boundary -> v = 0
        # so we want to be able to generate hyperplane for them to display them
        def hyperplane(x, w, b, v):
            return (-w[0] * x - b + v) / w[1]

        # setting limits for graph
        data_range = (self.min_feature_value * 0.9, self.max_feature_value * 1.1)
        hyp_x_min = data_range[0]
        hyp_x_max = data_range[1]

        # plotting hyperplanes using two points for a line in 2D
        # (w.x+b) = 1
        # positive support vector hyperplane
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)
        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')

        # negative support vector hyperplane
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

        # decision boundary hyperplane
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

        plt.show()


# data with kind of obvious separation of the two sections of data
data_dict = {-1: np.array([[1, 7], [2, 8], [3, 8]]),
             1: np.array([[5, 1], [6, -1], [7, 3]])}

clf = Support_Vector_Machine()
clf.fit(data=data_dict)

predict_us = [[0, 10], [1, 3], [3, 4], [3, 5], [5, 5], [5, 6], [5, -5], [5, 8]]
for p in predict_us:
    clf.predict(p)

clf.visualise()




