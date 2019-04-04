#!/usr/bin/env bash
python3 run_classifier.py --data_dir ../semeval/ --bert_model bert-large-uncased --task_name semeval --output_dir quicktest/ --max_seq_length 500 --do_train --do_lower_case --train_batch_size 32 --learning_rate 5e-6 --gradient_accumulation_steps 8 --eval_batch_size 4 --num_train_epochs 100 --model_path unsupervised_large_model/model.pth
