#!/usr/bin/env python3
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

#Notice: remove the leading space of each key!!!
def dataLoader(file_name, LSTM_data=False, Normalize=False):
        df = pd.read_csv(file_name)
        all_features = [
                        'label',
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
        df = df[all_features]
        X = df.drop('label', axis=1)
        Y = df['label']

        X = X.to_numpy()
        Y = Y.to_numpy()

        if LSTM_data:
                le = LabelEncoder()
                le.fit(Y)
                Y = le.transform(Y)
                # print(Y)
                le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                print("class mapping: ", le_name_mapping)
        
        if Normalize:
                scaler = StandardScaler()
                scaler.fit(X)
                X = scaler.transform(X)
        return X, Y

if __name__ == "__main__":
        X, Y = dataLoader(os.path.abspath('../datasets/train/shuffle.csv'))
        print(X)
        print(Y)
