import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42)
model = RandomForestClassifier(n_estimators=20,max_depth=3)
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
acc = accuracy_score(y_test, y_predict)
print("Accuracy is ", acc)


#if we use RF to predict the single data point, the output is the probability of two classes.
