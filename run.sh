./main.py --model DT --test_size 260 --random_state 15 --max_depth 1 --data_path datasets/train/training_mixed_loss_cong_100s.csv
./main.py --model RF --test_size 260 --random_state 42 --n_estimators 2  --max_depth 3 --data_path datasets/train/training_mixed_loss_cong_100s.csv
./main.py --model MLP --test_size 260 --random_state 42 --max_depth 3 --data_path datasets/train/training_mixed_loss_cong_100s.csv
# ./main.py --model LSTM --test_size 0.25 --random_state 50 --batch_size 5 --data_path datasets/train/training_mixed_loss_cong_100s.csv