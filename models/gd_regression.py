from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np


class GDRegression(ABC):
    def __init__(self, learning_rate=0.01, max_iterations=100):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = 0

    @abstractmethod
    def cost_function(self, X, y):
        pass

    def optimize(self, X, y):
        cost_func, theta_derivative, bias_derivative = self.cost_function(X, y)
        self.bias -= self.learning_rate * bias_derivative
        self.weights -= self.learning_rate * theta_derivative

        return cost_func

    def fit(self, X_train, y_train, X_val=None, y_val=None, metrics=None, report_periodicity=20):
        def calculate_metrics():
            print(f"\n{i}/{self.max_iterations}: Loss -> {np.round(cost_function, 3)}", end='')

            if any(v is None for v in (X_val, y_val, metrics)):
                return

            y_train_pred = self.predict(X_train)
            y_val_pred = self.predict(X_val)

            for metric in metrics:
                print(f", train {metric.__name__} -> {metric(y_train, y_train_pred)}"
                      f", val {metric.__name__} -> {metric(y_val, y_val_pred)}")

        if self.weights is None:
            self.weights = np.zeros(X_train.shape[1])

        cost_by_iteration = OrderedDict()

        for i in range(self.max_iterations + 1):
            cost_function = self.optimize(X_train, y_train)
            cost_by_iteration[i] = cost_function

            if i % report_periodicity == 0:
                calculate_metrics()

        return cost_by_iteration

    @abstractmethod
    def predict(self, X):
        pass
