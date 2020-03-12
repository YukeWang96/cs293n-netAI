#!/usr/bin/env python3
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus


class DT:
    def __init__(self, max_depth):
        print("=> [DT] Initialization, max_depth: {}".format(max_depth))
        self.model = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
        self.trained = False

    def train(self, features, labels, test_size=0.7, random_state=42, export=True):
        print("=> [DT] training, test size = {}, random_state = {}".format(test_size, random_state))
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)

        self.model.fit(X_train, y_train)
        y_predict = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, y_predict)
        print(len(X_train[0]))

        # sample test
        y_predict = self.model.predict(X_test)
        test_acc = accuracy_score(y_test, y_predict)
        pr_value = precision_recall_fscore_support(y_test, y_predict, labels=['cong', 'loss'])
        
        print("=> DT training, \
                  train_acc: {}, \
                  test_acc: {} ".format(train_acc, test_acc))
        
        print('=> Per Label PRF Value')
        print("\tcong\t\tloss")
        for v, i in zip(pr_value, ['prec.', 'recall', 'f1', 'supp.']):
            print(i,"\t","{:.2f}".format(v[0]), "\t\t", "{:.2f}".format(v[1]))

        self.trained = True

        feature_cols = [
                        # 'bandwidth',
                        'latency',
                        'rtt',
                        'jitter',
                        'ipa',
                        'win_incs',
                        'win_decs',
                        'retrans',
                        'outoforders'
                        ]
        # tree_rules = export_text(self.model, feature_names=list(X_train))
        if export:
            # print(tree_rules)
            dot_data = StringIO()
            export_graphviz(self.model, out_file=dot_data,  
                            filled=True, rounded=True, special_characters=True, 
                            feature_names = feature_cols, class_names=['cong','loss'])
            graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
            graph.write_png('DT_tree_'+ str(feature_cols) + '.png')
            Image(graph.create_png())

    def eval(self, X):
        assert self.trained
        return self.model.predict(X)
