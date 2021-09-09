from abc import ABC, abstractmethod
from collections import OrderedDict
import numpy as np
from metrics import f1_score


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

    def fit(self, X_train, y_train, X_val, y_val):
        def print_metrics():
            y_train_pred = self.predict(X_train)
            train_f1 = f1_score(y_train, y_train_pred)

            y_val_pred = self.predict(X_val)
            val_f1 = f1_score(y_val, y_val_pred)

            print(f"{i}/{self.max_iterations}: Loss -> {round(cost_function, 3)}, train F1 -> {round(train_f1, 3)},"
                  f" val f1 -> {round(val_f1, 3)}")

        if self.weights is None:
            self.weights = np.zeros(X_train.shape[1])

        cost_by_iteration = OrderedDict()

        for i in range(self.max_iterations + 1):
            cost_function = self.optimize(X_train, y_train)
            cost_by_iteration[i] = cost_function

            if i % 20 == 0:
                print_metrics()

        return cost_by_iteration

    @abstractmethod
    def predict(self, X):
        pass
