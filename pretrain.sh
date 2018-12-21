#!/usr/bin/env bash
python3 unsupervised_pretraining.py --data_dir ../semeval/ --bert_model bert-large-uncased --task_name semeval --output_dir unsupervised_large_model/ --max_seq_length 60 --do_lower_case --train_batch_size 64 --gradient_accumulation_steps 1 --num_train_epochs 1
