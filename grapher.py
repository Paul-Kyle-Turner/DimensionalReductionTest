# grapher
# Paul Turner
# always assumes a fixed starting point[-][0] = 1
import pylab as py
import numpy as np


class grapher:
    def __init__(self):
        self.figure = py.figure()

    def slope_of_weights(self, weights):
        return -(weights[1] / weights[2])

    def intercept_of_weights(self, weights):
        return -(weights[0] / weights[2])

    def find_axis(self, points):
        x_max = 0
        y_max = 0
        x_min = 0
        y_min = 0
        for i in range(len(points)):
            if points[i][0] > x_max:
                x_max = points[i][0]
            if points[i][0] < x_min:
                x_min = points[i][0]
            if points[i][1] > y_max:
                y_max = points[i][1]
            if points[i][1] < y_min:
                y_min = points[i][1]
        return [x_min - 1, x_max + 1, y_min - 1, y_max + 1]

    def create_plot(self, points, y_class=None, weights=None, target_weights=None, plot_string=None, x_ax=None,
                    y_ax=None, iterations=None, axis_array=None):
        if plot_string == None:
            plot_string = "plot"
        if x_ax == None:
            x_ax = "x"
        if y_ax == None:
            y_ax = "y"
        if axis_array == None:
            axis_array = self.find_axis(points)

        plot = self.figure.add_subplot(111)
        plot.set_title(plot_string)
        plot.set_xlabel(x_ax)
        plot.set_ylabel(y_ax)

        if y_class != None:
            class_pos_x = []
            class_pos_y = []
            class_neg_x = []
            class_neg_y = []
            for i in range(len(points)):
                if y_class[i] == 1:
                    class_pos_x.append(points[i][1])
                    class_pos_y.append(points[i][2])
                else:
                    class_neg_x.append(points[i][1])
                    class_neg_y.append(points[i][2])

            plot.plot(class_neg_x, class_neg_y, 'bx', label='0\'s')
            plot.plot(class_pos_x, class_pos_y, 'r+', label='1\'s')
            plot.plot(class_neg_x, class_neg_y, 'bx', label='2\'s')
            plot.plot(class_neg_x, class_neg_y, 'bx', label='3\'s')
            plot.plot(class_neg_x, class_neg_y, 'bx', label='4\'s')
            plot.plot(class_neg_x, class_neg_y, 'bx', label='5\'s')
            plot.plot(class_neg_x, class_neg_y, 'bx', label='6\'s')
            plot.plot(class_neg_x, class_neg_y, 'bx', label='7\'s')
            plot.plot(class_neg_x, class_neg_y, 'bx', label='8\'s')
            plot.plot(class_neg_x, class_neg_y, 'bx', label='9\'s')
        else:
            x_point = []
            y_point = []
            for i in range(len(points)):
                x_point.append(points[i][1])
                y_point.append(points[i][2])
                plot.plot(x_point, y_point, '+', label='points')

        x = np.array([-100, 100])

        if target_weights != None:
            print("target print")
            slope_target = self.slope_of_weights(target_weights)
            intercept_target = self.intercept_of_weights(target_weights)
            plot.plot(x, slope_target * x + intercept_target, linewidth=2, c='b', label='f')

        if weights != None:
            print("print weights")
            slope_g = self.slope_of_weights(weights)
            intercept_g = self.intercept_of_weights(weights)
            if iterations == None:
                plot.plot(x, slope_g * x + intercept_g, linewidth=2, c='r', label='g')
            else:
                plot.plot(x, slope_g * x + intercept_g, linewidth=2, c='r', label='g : ' + str(iterations))

        plot.axis(axis_array)
        plot.legend(loc='upper right')

    def show(self):
        print "in show"
        py.show()

