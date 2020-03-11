#!/usr/bin/env python3
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

#Notice: remove the leading space of each key!!!
def dataLoader(file_name):
        df = pd.read_csv(file_name)
        df = df[['label', 'bandwidth','latency','rtt','jitter','ipa','retrans','outoforders']]
        X = df.drop('label', axis=1)
        Y = df['label']
        X = X.to_numpy()
        Y = Y.to_numpy()
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)
        return X, Y

if __name__ == "__main__":
        X, Y = dataLoader(os.path.abspath('../datasets/train/shuffle.csv'))
        print(X)
        print(Y)
