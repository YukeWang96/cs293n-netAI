import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support

class RF:
    def __init__(self, n_estimators=20, max_depth=3):
        print("RF Initialization, n_estimators: {}, max_depth: {}".format(n_estimators, max_depth))
        self.model = RandomForestClassifier(n_estimators=100, criterion='entropy', max_depth=3)
        self.trained = False

    def train(self, features, labels, test_size=0.5, random_state=42):
        print("RF training, test size = {}, random_state = {}".format(test_size, random_state))
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)
        self.model.fit(X_train, y_train)
        y_predict = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, y_predict)

        # sample test
        y_predict = self.model.predict(X_test)
        test_acc = accuracy_score(y_test, y_predict)
        pr_value = precision_recall_fscore_support(y_test, y_predict, labels=['loss', 'cong'])


        print("RF training, \
                train_acc: {}, \
                test_acc: {}".format(train_acc, test_acc))

        print('=> Per Label PRF Value')
        print("\tloss\t\tcong")
        for v, i in zip(pr_value, ['prec.', 'recall', 'f1', 'sup']):
            print(i,"\t","{:.2f}".format(v[0]), "\t\t","{:.2f}".format(v[1]))

        self.trained = True


    def eval(self, X):
        assert self.trained
        return self.model.predict(X)



# 
# print("Accuracy is ", acc)
#if we use RF to predict the single data point, the output is the probability of two classes.
