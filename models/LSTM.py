#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class netDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return (self.features[idx], self.labels[idx])

class LSTMTagger(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMTagger, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, output_dim)

    def forward(self, features):
        # print("feature size: ", features.size())
        # features = features.double()
        lstm_out, _ = self.lstm(features.view(len(features), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(features), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores


class LSTM:
    def __init__(self, input_dim = 4, hidden_dim = 3, output_dim=3, lr=0.1):
        print("LSTM Initialization, input_dim: {}, hidden_dim: {}, output_dim: {}".format(input_dim, hidden_dim, output_dim))
        self.model = LSTMTagger(input_dim, hidden_dim, output_dim).double()
        self.loss_function = nn.NLLLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.1)
        self.trained = False
        self.train_acc = []
        self.test_acc = []

    def train(self, features, labels, test_size=0.7, random_state=42, epoch=300, batch=5):
        print("LSTM training, test size = {}, random_state = {}".format(test_size, random_state))
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)
        
        netData_train = netDataset(X_train, y_train)
        loader_train = DataLoader(netData_train, batch_size=batch)

        netData_test = netDataset(X_test, y_test)
        loader_test = DataLoader(netData_test, batch_size=batch)

        for epoch in range(epoch):
            self.model.zero_grad()
            cnt = 0
            self.train_acc = []

            for batch_train in loader_train:
                X, y = batch_train[0], batch_train[1]
                tag_scores = self.model(X)
                
                _, train_label = torch.max(tag_scores, dim=1)
                acc = accuracy_score(train_label, y)
                self.train_acc.append(acc)
                self.loss = self.loss_function(tag_scores, y)
                self.loss.backward()
                self.optimizer.step()
                print("Batch: {}/{}, loss: {}".format(cnt, len(loader_train), self.loss))
                cnt += 1
                
            print("==> Epoch: {}, accuracy: {}".format(epoch, sum(self.train_acc)/len(self.train_acc)))


        # test evaluation
        for batch_test in loader_test: 
            X, y = batch_test[0], batch_test[1]
            tag_scores = self.model(X)
            _, test_label = torch.max(tag_scores, dim=1)
            acc = accuracy_score(test_label, y)
            self.test_acc.append(acc)

        print("LSTM training, \
                train_acc: {}, \
                test_acc: {}".format(sum(self.train_acc)/len(self.train_acc), sum(self.test_acc)/len(self.test_acc)))

        self.trained = True


    # def eval(self, X):
    #     assert self.trained
    #     with torch.no_grad():
    #         tag_scores = model(X)
