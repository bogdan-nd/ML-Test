import numpy as np

from models.gd_regression import GDRegression


def sigmoid(z):
    """
    Calculates sigmoid function
    """
    return 1 / (1 + np.exp(-z))


class MyLogisticRegression(GDRegression):
    """
    Logistic Regression Classifier
    """

    def __init__(self, learning_rate=0.01, max_iterations=100):
        super().__init__(learning_rate, max_iterations)

    def cost_function(self, X, y):
        m = X.shape[0]

        h_x = sigmoid(np.dot(X, self.weights) + self.bias)
        h_x = h_x.reshape(-1)
        cost_function = (-1 / m) * np.sum(y * np.log(h_x) + (1 - y) * np.log(1 - h_x))

        bias_derivative = (1 / m) * np.sum(h_x - y)
        theta_derivative = (1 / m) * np.dot(X.T, h_x - y)

        return cost_function, theta_derivative, bias_derivative

    def predict(self, X):
        probability = sigmoid(np.dot(X, self.weights) + self.bias)
        return np.array([1 if prob >= 0.5 else 0 for prob in probability])
