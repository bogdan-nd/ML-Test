from abc import ABC, abstractmethod
from collections import OrderedDict

import numpy as np


class GDRegression(ABC):
    """
    Base class for Regression models with Gradient Descent optimizer
    """

    def __init__(self, learning_rate=0.01, max_iterations=100):
        """
        :param learning_rate: learning model of the model
        :param max_iterations: the maximum number of iterations in the 'fit' method
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = 0

    @abstractmethod
    def cost_function(self, X, y):
        """
        Calculates the cost function and the gradient

        :param X: matrix of (m,n) shape, which have m samples with n features
        :param y: target vector of m shape
        :return: cost function, gradient of parameter vector and bias gradient
        """
        pass

    def optimize(self, X, y):
        """
        Updates the bias and the weights

        :param X: matrix of (m,n) shape, which have m samples with n features
        :param y: target vector of m shape
        :return: cost function
        """
        cost_func, theta_derivative, bias_derivative = self.cost_function(X, y)
        self.bias -= self.learning_rate * bias_derivative
        self.weights -= self.learning_rate * theta_derivative

        return cost_func

    def fit(self, X_train, y_train, X_val=None, y_val=None, metrics=None, report_periodicity=20):
        """
        Fits the model, calculates and prints specific metrics of each iteration

        :param X_train: matrix of (m1,n1) shape, which have m1 samples with n1 features, destined for training
        :param y_train: target vector of m1 shape, destined for training
        :param X_val: matrix of (m2,n2) shape, which have m2 samples with n2 features, destined for cross-validation
        :param y_val: target vector of m2 shape, destined for cross-validation
        :param metrics: specific metrics to calculate
        :param report_periodicity: the sequence number for calculating and printing specific metrics
        :return: dictionary of cost function by iteration number
        """

        def calculate_metrics():
            """
            Calculates specific metrics

            :return: None
            """
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
        """
        Predicts the target for samples in X

        :param X: matrix of (m,n) shape, which have m samples with n features, destined for training
        :return: target prediction
        """
        pass
