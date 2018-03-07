# menu system for classifier

import numpy as np
import random
import generator as gener
from cvxopt import solvers


class classifier():
    def __init__(self, rand=None, dim=2, threshold=0):
        gen = gener.generator()
        self.dim = dim
        self.threshold = 0
        if rand == None:
            self.weights = gen.create_weights(self.dim)
        else:
            self.weights = gen.create_random_weights(self.dim)

    def get_weights(self):
        return self.weights

    def update_weight(self, vector, y, learning_rate=None):
        if learning_rate == None:
            for i in range(len(self.weights)):
                self.weights[i] += y * vector[i]
        else:
            for i in range(len(self.weights)):
                self.weights[i] += learning_rate * (y - self.weights[i] * vector[i]) * vetor[i]

    def classify_one_point(self, vector, weights=None):
        if weights == None:
            return 1 if (np.inner(vector, self.weights)) >= self.threshold else -1
        else:
            return 1 if np.inner(vector, weights) >= self.threshold else -1

    def classify_all_points(self, points):
        y_class = []
        for row in range(len(points)):
            if self.classify_one_point(points[row]) == -1:
                y_class.append(-1)
            else:
                y_class.append(1)
        return y_class

    def check_error(self, points, y_class, weights=None):
        error = 0.0
        for i in range(len(points)):
            if weights == None:
                if y_class[i] != self.classify_one_point(points[i]):
                    error += 1
            else:
                if y_class[i] != self.classify_one_point(points[i], weights):
                    error += 1
        error = error / len(points)
        return error

    def linear_regression_train(self, points, y_class):
        self.weights = np.dot(np.linalg.pinv(points), y_class)
        return self.weights, self.check_error(points, y_class)

    def missclass_points(self, points, y_class):
        missclass = []
        y_class_hat = []
        for i in range(len(points)):
            if self.classify_one_point(points[i]) != y_class[i]:
                missclass.append(points[i])
                y_class_hat.append(y_class[i])
        return missclass, y_class_hat

    def pocket_train(self, points, y_class, iterations=1000, laerning_rate=None, rand=None):
        error = self.check_error(points, y_class)
        weight_hat = self.weights
        error_iter = []
        for i in range(iterations):
            if i % 50000 == 0:
                print(i)
            missclass, y_class_hat = self.missclass_points(points, y_class)
            if len(missclass) != 0:
                self.update_weight(missclass[1], y_class_hat[1])
                error_hat = self.check_error(points, y_class)
                if error > error_hat:
                    error = error_hat
                    weight_hat = self.weights
                    error_iter.append((1, error, i))
            else:
                return error, weight_hat, error_iter
        return error, weight_hat, error_iter

    def pocket_train_transform(self, points, y_class, iterations=1000, laerning_rate=None, rand=None):
        error = self.check_error(points, y_class)
        weight_hat = self.weights
        error_iter = []
        for i in range(iterations):
            if i % 50000 == 0:
                print(i)
            missclass, y_class_hat = self.missclass_points(points, y_class)
            if len(missclass) != 0:
                self.update_weight(missclass[1], y_class_hat[1])
                error_hat = self.check_error(points, y_class)
                if error > error_hat:
                    error = error_hat
                    weight_hat = self.weights
                    error_iter.append((1, error, i))
            else:
                return error, weight_hat, error_iter
        return error, weight_hat, error_iter

    def pla_train(self, points, y_class, iterations=1000, learning_rate=None, rand=None):

        con_iter = 0

        if rand != None:
            idx = np.random.randint(len(points), size=len(points))

        for it in range(iterations):
            con_iter += 1
            error = False
            for row in range(len(points)):
                classification = None
                if rand != None:
                    classification = self.classify_one_point(points[idx[row]])
                else:
                    classification = self.classify_one_point(points[row])
                if classification != y_class[row]:
                    self.update_weight(points[row], y_class[row], learning_rate)
                    error = True
            if not error:
                return con_iter, self.weights
        return con_iter, self.weights

    def test(self, points, y_class=None, number_of_test_points=1000, target_weights=None, max_it_train=10000,
             learning_rate=None, rand=None):
        error = 0.0

        if y_class == None and target_weights == None:
            print "test fail, give classifier"

        for t in range(number_of_test_points):
            test_point = [1]
            for i in range(self.dim + 1):
                test_point.append(np.random.uniform(-1, 1))
                if target_weights == None:
                    e_out = self.classify_one_point(test_point, target_weights)
                else:
                    e_out = y_class[i]
                e_in = self.classify_one_point(test_point, self.weights)
            if e_out != e_in:
                error += 1
        return error / number_of_tests

    def multiple_test(self, points, y_class=None, target_weights=None, number_of_points_per_test=None,
                      number_of_tests=None, max_it_train=None, learning_rate=None, rand=None):

        if y_class == None and target_weights == None:
            print"you broke me"

        if number_of_tests == None:
            number_of_tests = 100
        if number_of_points_per_test == None:
            number_of_points_per_test = 1000
        if max_it_train == None:
            max_it_train = 1000

        error = []

        for i in range(number_of_tests):
            error.append(
                self.test(points, y_class, number_of_points_per_test, target_weights, max_it_train, learning_rate,
                          rand))

        return error

    def QPsolve(self, Qmatrix, pmatrix, Amatrix, cmatrix, weights=None):
        if weights == None:
            self.weights = solvers.qp(Qmatrix, pmatrix, Amatrix, cmatrix)
            weights = self.weights
        else:
            weights = solvers.qp(Qmatrix, pmatrix, Amatrix, cmatrix)
        return weights



