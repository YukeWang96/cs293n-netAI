#!/usr/bin/env python3
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris

class MLP:
    def __init__(self, max_depth=3):
        print("=> MLP Initialization, max_depth: {}".format(max_depth))
        self.model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(32, 16))
        self.trained = False

    def train(self, features, labels, test_size=0.7, random_state=42):
        print("=> MLP training, test size = {}, random_state = {}".format(test_size, random_state))
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)

        self.model.fit(X_train, y_train)
        y_predict = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, y_predict)

        # sample test
        y_predict = self.model.predict(X_test)
        test_acc = accuracy_score(y_test, y_predict)

        print("=> MLP training, \
                train_acc: {}, \
                test_acc: {}".format(train_acc, test_acc))

        self.trained = True


    def eval(self, X):
        assert self.trained
        return self.model.predict(X)

# X, y = load_iris(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)
# model.fit(X_train, y_train)
# y_predict = model.predict(X_test)
# acc = accuracy_score(y_test, y_predict)
# print("Accuracy is ", acc)