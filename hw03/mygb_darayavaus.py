#coding=utf-8

from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
import numpy as np


# Ваш email, который вы укажете в форме для сдачи
AUTHOR_EMAIL = 'darayavaus.chern@yandex.ru'
# Параметрами с которыми вы хотите обучать деревья
TREE_PARAMS_DICT = {'max_depth': 4}
# Параметр tau (learning_rate) для вашего GB
TAU = 0.1


class SimpleGB(BaseEstimator):
    def __init__(self, tree_params_dict, iters, tau):
        self.tree_params_dict = tree_params_dict
        self.iters = iters
        self.tau = tau
       
    @staticmethod
    def sigmoid(x):
        return 1. / (1 + np.exp(-x))

    @staticmethod
    def der_sigmoid(x):
        return np.exp(-x) / (1 + np.exp(-x)) ** 2
    
    @staticmethod
    def der_log_loss(y_hat, y_true):
        return y_true / y_hat - (1 - y_true) / (1 - y_hat)
    
    def calc_gradient(self, y_hat, y_true):
        return self.der_log_loss(self.sigmoid(y_hat), y_true) * self.der_sigmoid(y_hat)
        
    def fit(self, X_data, y_data):
        self.base_algo = DecisionTreeRegressor(**self.tree_params_dict).fit(X_data, y_data)
        self.estimators = []
        curr_pred = self.base_algo.predict(X_data)
        for iter_num in range(self.iters):
            # Градиент функции потерь
#             grad = y_data / (1 + np.exp(-y_data*curr_pred))
            grad = self.calc_gradient(curr_pred, y_data)
            # Обучаем DecisionTreeRegressor предсказывать антиградиент
            algo = DecisionTreeRegressor(**self.tree_params_dict).fit(X_data, grad)

            self.estimators.append(algo)
            # Обновляем предсказания в каждой точке
            curr_pred += self.tau * algo.predict(X_data)
        return self
    
    def predict(self, X_data):
        # Предсказание на данных
        res = self.base_algo.predict(X_data)
        for estimator in self.estimators:
            res += self.tau * estimator.predict(X_data)
        # Задача классификации, поэтому надо отдавать 0 и 1
        return res > 0.
