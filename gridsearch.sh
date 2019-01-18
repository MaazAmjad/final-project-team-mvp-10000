#!/usr/bin/env bash
python3 grid_search.py bert-large-uncased gridsearchtest --max_seq_len 10 13 1 --learning_rate 1e-5 2e-5 1e-5 --num_train_epochs 1 2 1
