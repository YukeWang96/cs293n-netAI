#!/usr/bin/env python3
import subprocess

# front = "./main.py --model RF --test_size 0.3 --random_state 42 --n_estimators".split() 
  
# tail = "--max_depth 3 --data_path datasets/train/training_mixed_loss_cong_100s.csv".split()

# for i in range(1, 100):
#     all = front + [str(i)] + tail
#     subprocess.run(all)


DT = "./main.py --model DT --test_size 0.3 --random_state 42 --max_depth 3 --data_path datasets/train/training_mixed_loss_cong_100s.csv > DT.log".split()


RF = "./main.py --model RF --test_size 0.3 --random_state 42 --n_estimators 3 --max_depth 3 --data_path datasets/train/training_mixed_loss_cong_100s.csv > RF.log".split()


MLP = "./main.py --model MLP --test_size 0.3 --random_state 42 --data_path datasets/train/training_mixed_loss_cong_100s.csv > MLP.log".split()

LSTM = "./main.py --model LSTM --test_size 0.25 --random_state 50 --batch_size 5 --data_path datasets/train/training_mixed_loss_cong_100s.csv".split()

subprocess.run(DT)
subprocess.run(RF)
subprocess.run(MLP)