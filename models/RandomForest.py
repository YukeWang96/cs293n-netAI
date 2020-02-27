import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class RF:
    def __init__(self. n_estimators=20, max_depth=3):
        print("RF Initialization, \
                n_estimators: {}, \
                max_depth: {}".format(n_estimators. max_depth))
        self.model = RandomForestClassifier(n_estimators, max_depth)
        self.trained = False

    def train(self, features, labels, test_size=0.50, random_state=42):
        print("RF training, \
                test size = {}, \
                random_state = {}".format(test_size, random_state))
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size, random_state)

        self.model.fit(X_train, y_train)
        y_predict = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, y_predict)

        # sample test
        y_predict = self.model.predict(X_test)
        test_acc = accuracy_score(y_test, y_predict)

        print("RF training, \
                train_acc: {}, \
                test_acc: {}".format(train_acc, test_acc))

        self.trained = True


    def eval(self, X):
        assert self.trained
        return self.model.predict(X)



# 
# print("Accuracy is ", acc)
#if we use RF to predict the single data point, the output is the probability of two classes.
