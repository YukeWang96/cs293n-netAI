import pandas as pd

#Notice: remove the leading space of each key!!!
def dataLoader(file_name):
        df = pd.read_csv(file_name)
        df = df[['bandwidth','latency','rtt','jitter','ipa','retrans','outoforders','label']]
        X = df.drop('label', axis=1)
        Y = df['label']
        return X, Y

X,Y = dataLoader('training_loss50_10s.csv')
print(X)
print(Y)
