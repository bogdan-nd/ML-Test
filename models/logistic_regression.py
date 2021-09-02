import numpy as np
from collections import OrderedDict
from metrics import f1_score


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class MyLogisticRegression:
    def __init__(self, learning_rate=0.01, max_iterations=100):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = 0

    def cost_function(self, X, y):
        m = X.shape[0]

        h_x = sigmoid(np.dot(X, self.weights) + self.bias)
        h_x = h_x.reshape(-1)
        cost_function = (-1 / m) * np.sum(y * np.log(h_x) + (1 - y) * np.log(1 - h_x))

        bias_derivative = (1 / m) * np.sum(h_x - y)
        theta_derivative = (1 / m) * np.dot(X.T, h_x - y)

        return cost_function, theta_derivative, bias_derivative

    def optimize(self, X, y):
        cost_func, theta_derivative, bias_derivative = self.cost_function(X, y)
        self.bias -= self.learning_rate * bias_derivative
        self.weights -= self.learning_rate * theta_derivative

        return cost_func

    def fit(self, X_train, y_train, X_val, y_val):
        if self.weights is None:
            self.weights = np.zeros(X_train.shape[1])

        cost_by_iteration = OrderedDict()

        for i in range(self.max_iterations+1):
            cost_function = self.optimize(X_train, y_train)
            cost_by_iteration[i] = cost_function

            if i % 50 == 0:
                y_train_pred = self.predict(X_train)
                train_f1 = f1_score(y_train, y_train_pred)

                y_val_pred = self.predict(X_val)
                val_f1 = f1_score(y_val, y_val_pred)

                print(f"{i}/{self.max_iterations}: Loss -> {round(cost_function, 3)}, train F1 -> {round(train_f1, 3)},"
                      f" val f1 -> {round(val_f1, 3)}")

        return cost_by_iteration

    def predict(self, X):
        probability = self.predict_proba(X)
        return np.array([1 if prob >= 0.5 else 0 for prob in probability])

    def predict_proba(self, X):
        return sigmoid(np.dot(X, self.weights) + self.bias)
