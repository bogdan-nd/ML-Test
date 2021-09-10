import numpy as np

from models.gd_regression import GDRegression


class MyLinearRegression(GDRegression):
    """
    Linear Regression with L2 regularization
    """

    def __init__(self, learning_rate=0.01, max_iterations=100, regularization_term=0.1):
        super().__init__(learning_rate, max_iterations)
        self.regularization_term = regularization_term

    def cost_function(self, X, y):
        m = X.shape[0]

        h_x = np.dot(X, self.weights) + self.bias
        h_x = h_x.reshape(-1)
        cost_function = (1 / (2 * m)) * (np.sum((h_x - y) ** 2) + self.regularization_term * np.sum(self.weights ** 2))

        bias_derivative = (1 / m) * np.sum(h_x - y)
        theta_derivative = (1 / m) * (np.dot(X.T, h_x - y) + self.regularization_term * self.weights)

        return cost_function, theta_derivative, bias_derivative

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
