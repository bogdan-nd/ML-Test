import numpy as np

from models.gd_regression import GDRegression


class MyLinearRegression(GDRegression):
    def __init__(self, learning_rate=0.01, max_iterations=100):
        super().__init__(learning_rate, max_iterations)

    def cost_function(self, X, y):
        m = X.shape[0]

        h_x = np.dot(X, self.weights) + self.bias
        cost_function = (1 / (2 * m)) * (h_x - y) ** 2

        bias_derivative = (1 / m) * (h_x - y)
        theta_derivative = (1 / m) * np.dot(X.T, h_x - y)

        return cost_function, theta_derivative, bias_derivative

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
