#!/usr/bin/env python3

import argparse
from models.RandomForest import RF
from models.DecisionTree import DT
from models.MLP import MLP
from models.LSTM import LSTM

from sklearn.datasets import load_iris


# parse input argument
parser = argparse.ArgumentParser(description='packet loss classifier')
parser.add_argument('--model', type=str, default=None, help='Models for classifier [DNN, LSTM, RandomForest(RF), DecisionTree(DT)]')
parser.add_argument('--epoch', type=int, help='training epochs for LSTM')
parser.add_argument('--random_state', type=int, default=0, help='random state for [DNN, RF, DF] train-test data split')
parser.add_argument('--test_size', type=float, default=0.25, help='test dataset size, default = 0.25')
parser.add_argument('--n_estimators', type=int, default=20, help='number of RF estimators, default=20')
parser.add_argument('--max_depth', type=int, default=3, help='maximum depth of [RF, DT] model, default=3')
args = parser.parse_args()

classifier = {
    "RF": RF(),
    "DT": DT(),
    "MLP": MLP(),
    "LSTM": LSTM()
}

if __name__ == "__main__":
    assert args.model != None and args.model in classifier
    model = classifier[args.model]
    X, y = load_iris(return_X_y=True)
    model.train(X, y)