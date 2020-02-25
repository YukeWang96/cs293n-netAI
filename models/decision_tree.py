#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)

model = DecisionTreeClassifier(criterion='entropy', splitter='best', random_state=0)
model.fit(X_train, y_train)
X_pred = model.predict(X_test)
print("Accuracy: {}%".format(sum(X_pred == y_test)/len(X_pred) * 100))