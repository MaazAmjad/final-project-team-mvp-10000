#!/usr/bin/env bash
python3 unsupervised_pretraining.py --data_dir semeval/ --bert_model bert-base-uncased --task_name semeval --output_dir unsupervised_model/ --max_seq_length 60 --do_lower_case --train_batch_size 256 --gradient_accumulation_steps 1 --num_train_epochs 1
