# point creator

import numpy as np
import random
import classifier


class generator:
    def __init__(self, n=None, dim=None):
        self.set_dim(dim)
        self.set_n(n)

    def set_dim(self, dim=2):
        self.dim = dim

    def set_n(self, n=10000):
        self.n = n

    def create_random_weights(self, dim=2):
        weights = np.random.rand(dim + 1)
        return weights

    def create_weights(self, dim=2):
        weights = np.zeros(dim + 1)
        return weights

    def create_test_point(self, target_weights, w_min, w_max, dim=None):
        vector = np.ones((1, dim + 1))
        for i in range(dim):
            vector[i + 1] = random.uniform()

    def create_training_points(self, n=None, dim=None, w_min=None, w_max=None, classifier=None):
        if n == None:
            n = 10000
        if dim == None:
            dim = 2
        if w_min == None:
            w_min = -1
        if w_max == None:
            w_max = 1
        print
        points = np.ones((n, dim + 1))
        for i in range(n):
            for k in range(dim):
                points[i][k + 1] = random.uniform(w_min, w_max)
        if classifier == None:
            return points
        else:
            y_class = classifier.classify_all_points(points)
            return points, y_class

    def create_semi_circle_points(self, rad=None, thick=None, sep=None, x_shift=None, n=None, dim=None):
        if rad == None:
            rad = 1
        if sep == None:
            sep = 1
        if thick == None:
            thick = 1
        if x_shift == None:
            x_shift = 1
        if n == None:
            n = 10000
        if dim == None:
            dim = 2

        points = np.ones((n, dim + 1))
        y_class = []
        use = True
        for i in range(n):
            theta = np.deg2rad(random.uniform(0, 1) * 180)
            radT = rad + (random.uniform(0, 1) * thick)
            x = radT * np.cos(theta)
            y = ((radT * np.sin(theta)) + sep)
            if use:
                x = -(x) + x_shift
                y = -(y)
                y_class.append(-1)
                use = False
            else:
                y_class.append(1)
                use = True
            points[i][1] = x
            points[i][2] = y
        return points, y_class

    def d2_to_transform(self, points):
        points_t = []
        for row in range(points):
            vector = []
            vector.append(1)
            vector.append(points[row][1])
            vector.append(points[row][2])
            vector.append(points[row][1] * points[row][1])
            vector.append(points[row][1] * points[row][2])
            vector.append(points[row][2] * points[row][2])
            points_t.append(vector)
        return points_t

    def dim_num_transform(self, n = None):
        if n == None :
            n = 2
        else :
            #TODO create number Q(Q+3)/Q
            print("do")

    def createQPmatrixDual(self, points, y_class):
        dataMatrix = []
        for i in range(len(points)):
            vector = []
            for k in range(len(points.T)):
                vector.append(points[i][k] * y_class[i])

            dataMatrix.append(vector)

        dataMatrixTranspose = np.transpose(dataMatrix)
        Q = np.dot(dataMatrix, dataMatrixTranspose)

        A = []
        A.append(y_class)
        A.append(y_class * -1)
        for i in range(len(y_class)):
            vector = []
            for k in range(len(y_class)):
                if i == k:
                    vector.append(1)
                else:
                    vector.append(0)
            A.append(vector)

        p = []
        for i in range(len(points)):
            p.append(-1)

        c = []
        for k in range(len(points) + 2):
            c.append(0)

        return Q, p, A, c
