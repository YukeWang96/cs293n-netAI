#!/usr/bin/env python3
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  
import matplotlib.pyplot as plt 
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(32, 16))
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
acc = accuracy_score(y_test, y_predict)
print("Accuracy is ", acc)
