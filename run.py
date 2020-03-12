#!/usr/bin/env python3
import subprocess

# front = "./main.py --model RF --test_size 0.3 --random_state 42 --n_estimators".split() 
  
# tail = "--max_depth 3 --data_path datasets/train/training_mixed_loss_cong_100s.csv".split()

# for i in range(1, 100):
#     all = front + [str(i)] + tail
#     subprocess.run(all)


# DT = "./main.py --model DT --test_size 0.3 --random_state 42 --max_depth 3 --data_path datasets/train/training_mixed_loss_cong_100s.csv".split()
# subprocess.run(DT)

# RF = "./main.py --model RF --test_size 0.3 --random_state 42 --n_estimators 3 --max_depth 3 --data_path datasets/train/training_mixed_loss_cong_100s.csv".split()
# subprocess.run(RF)

# MLP = "./main.py --model MLP --test_size 0.3 --random_state 42 --data_path datasets/train/training_mixed_loss_cong_100s.csv".split()
# subprocess.run(MLP)

LSTM = "./main.py --model LSTM --test_size 0.25 --epoch 20 --random_state 50 --batch_size 10 --input_dim 8 --hidden_dim 3 --output_dim 2 --data_path datasets/train/training_mixed_loss_cong_100s.csv".split()
subprocess.run(LSTM)