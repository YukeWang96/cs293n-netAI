#!/usr/bin/env python3

import argparse
from models.dataLoader import *
from models.RandomForest import RF
from models.DecisionTree import DT
from models.MLP import MLP
from models.LSTM import LSTM

# from sklearn.datasets import load_iris


# parse input argument
parser = argparse.ArgumentParser(description='packet loss classifier')
parser.add_argument('--model', type=str, default=None, help='Models for classifier [DNN, LSTM, RandomForest(RF), DecisionTree(DT)]')
parser.add_argument('--random_state', type=int, default=0, help='random state for [MLP, RF, DF] train-test data split')
parser.add_argument('--test_size', type=int, default=0.25, help='test dataset size, default = 0.25')
parser.add_argument('--data_path', type=str, help='test dataset size, default = 0.25')

parser.add_argument('--n_estimators', type=int, default=20, help='number of [RF] estimators, default=20')
parser.add_argument('--max_depth', type=int, default=3, help='maximum depth of [RF, DT] model, default=3')

parser.add_argument('--input_dim', type=int, default=8, help='input_dim of [LSTM] model, default=3')
parser.add_argument('--hidden_dim', type=int, default=3, help='hidden_dim of [LSTM] model, default=3')
parser.add_argument('--output_dim', type=int, default=3, help='output_dim of [LSTM] model, default=3')
parser.add_argument('--epoch', type=int, default=300, help='epoch of [LSTM] model, default=300')
parser.add_argument('--batch_size', type=int, default=5, help='batch size of [LSTM] model, default=5')
args = parser.parse_args()

if __name__ == "__main__":
    assert args.model != None and \
        args.model in ["RF", "DT", "MLP", "LSTM"]

    # model initialization
    model = None
    if args.model == "RF": 
        model = RF(n_estimators=args.n_estimators, max_depth=args.max_depth)
    if args.model == "DT": 
        model = DT(max_depth=args.max_depth)
    if args.model == "MLP": 
        model = MLP(max_depth=args.max_depth)
    if args.model == "LSTM": 
        model = LSTM(input_dim = 4, hidden_dim = 3, output_dim=3)

    # data loading 
    # X, y = load_iris(return_X_y=True)
    X, y = dataLoader(args.data_path)

    # model training
    if args.model == "LSTM":
        model.train(X, y, test_size=args.test_size, random_state=args.random_state, epoch=args.epoch, batch=args.batch_size)
    else:
        model.train(X, y, test_size=args.test_size, random_state=args.random_state)
    
    print()
    print()