# coding=utf-8

import subprocess
import argparse
import itertools
import numpy as np

def create_directory_name(bert_model, max_seq_len, learning_rate, num_train_epochs):
    return f'{bert_model}_{max_seq_len}_{learning_rate}_{num_train_epochs}'

def train_model(output_grid_dir_name, bert_model, max_seq_len, learning_rate, num_train_epochs):
    output_directory = create_directory_name(bert_model, max_seq_len, learning_rate, num_train_epochs)
    model_arguments = '--output_dir=' + output_grid_dir_name + '/' + output_directory + '/ --max_seq_length=' + str(max_seq_len) + ' --learning_rate=' + str(learning_rate) + ' --num_train_epochs=' + str(num_train_epochs) 
    shared_arguments = '--data_dir=../semeval/ --bert_model=bert-large-uncased --task_name=semeval --do_train --do_lower_case --train_batch_size=32 --gradient_accumulation_steps=8 --eval_batch_size=4 --model_path=unsupervised_large_model/model.pth --model_no_save'
    
    print('python3 run_classifier.py ' + model_arguments + ' ' + shared_arguments) 
    subprocess.run(['python3', 'run_classifier.py'] + model_arguments.split() + shared_arguments.split())
    
def main():
    parser = argparse.ArgumentParser(description='Perform hyperparameter search on a model (specified with the underlying --model_path flag) by tweaking maximum sequence length, learning rate, and training epochs. Every parameter takes 3 arguments: the lower bound, the upper bound, and the step size. Models are placed in the grid_search/ directory.')
    parser.add_argument('bert_model', help='The string representing your choice of BERT model.')
    parser.add_argument('output_grid_dir_name', help='The name of the directory where the grid search should be stored.')
    parser.add_argument('--max_seq_len', nargs=3, type=int, help='Two endpoints and step size for maximum sequence length used by BERT model.')
    parser.add_argument('--learning_rate', nargs=3, type=float, help='Two endpoints and step size for learning rate used when training BERT model.')
    parser.add_argument('--num_train_epochs', nargs=3, type=int, help='Two endpoints and steop size for number of epochs for which the BERT model should train.')
    parser.add_argument('--checkpoint', type=int, help='The model number from which to continue training from. Models begin indexing from 0 because they represent an iteration of the for loop. This flag is useful for continuing training models if a grid search is interrupted')
    opt = parser.parse_args()
    
    # Train all combinations of models
    print('Arguments:', opt)
    parameters = [np.arange(*opt.max_seq_len), np.arange(*opt.learning_rate), np.arange(*opt.num_train_epochs)] 
    print('Ranges (max_seq_len, learning_rate, num_train_epochs):', parameters)
    param_combinations = list(itertools.product(*parameters))
    num_param_combinations = len(list(param_combinations))
    accept = input('You will need to train %d models. Are you sure you\'d like to continue? (y/n) ' % num_param_combinations)
    if accept != 'y':
        return

    for model_number, params in enumerate(param_combinations):
        if opt.checkpoint and model_number < opt.checkpoint:
            # A checkpoint continues an interrupted grid search
            # We want to skip over models that have already trained
            continue

        print('\nTraining model', model_number, 'of', num_param_combinations, 'with params', params)
        train_model(opt.output_grid_dir_name, opt.bert_model, *params)


if __name__ == '__main__':
    main()
