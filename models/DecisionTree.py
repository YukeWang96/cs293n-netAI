#!/usr/bin/env python3
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

class DT:
    def __init__(self, max_depth=3):
        print("=> [DT] Initialization, max_depth: {}".format(max_depth))
        self.model = DecisionTreeClassifier(criterion='entropy', max_depth=3)
        self.trained = False

    def train(self, features, labels, test_size=0.7, random_state=42):
        print("=> [DT] training, test size = {}, random_state = {}".format(test_size, random_state))
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)

        self.model.fit(X_train, y_train)
        y_predict = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, y_predict)

        # sample test
        y_predict = self.model.predict(X_test)
        test_acc = accuracy_score(y_test, y_predict)

        print("=> DT training, \
                  train_acc: {}, \
                  test_acc: {}".format(train_acc, test_acc))

        self.trained = True


    def eval(self, X):
        assert self.trained
        return self.model.predict(X)
