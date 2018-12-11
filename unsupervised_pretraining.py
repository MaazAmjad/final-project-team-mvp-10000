# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
import math
import multiprocessing
import spacy
from pathlib import Path
from itertools import count, repeat
from tqdm import tqdm, trange
import re

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForPreTraining
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
nlp_basic = spacy.load('en', pipeline=[])
wordRE = re.compile(r'\w')

class InputExample(object):
    """A single training example for unsupervised training."""

    def __init__(self, guid, text_a, text_b, next_sentence_label):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            next_sentence: boolean. Indicates if text_b is the next sentence or not.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.next_sentence_label = next_sentence_label

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()


class PretrainingDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

     def __len__(self):
        return len(self.examples)

     def __getitem__(self, idx):

def extract_sentences(article):
    return article.split('-eos-')

def extract_examples(datapoint):
    i, text = datapoint

    sentences = extract_sentences(text)
    sentence_count = len(sentences)
    text_examples = []

    if sentence_count <= 2:
        return text_examples

    for j, sentence in enumerate(sentences[:-1]):
        text_a = sentence.strip()
        index_b, next_sentence_label = next_sentence(j, sentence_count)
        text_b = sentences[index_b].strip()
        guid = "%d-%d" % (i, j)
        text_examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, next_sentence_label=next_sentence_label))

    return text_examples

def next_sentence(index, sentence_count):
    if random.random() <= 0.5:
        return index+1, 0
    else:
        if random.random() <= index / (sentence_count-2):
            guess = random.randrange(0, index)
        else:
            guess = random.randrange(index+2, sentence_count)
        
        return guess, 1

class SemevalProcessor(DataProcessor):
    """Processor for the Semeval data set."""
    def get_train_examples(self, data_dir):
        train_directory = os.path.join(data_dir, "training/preprocessed")
        data_file = open(os.path.join(train_directory, "articles-training-bypublisher-20181122.prep.txt"), "r")

        examples = []

        p = multiprocessing.Pool(multiprocessing.cpu_count())

        for text_examples in tqdm(p.imap(extract_examples, zip(count(), data_file), chunksize=100), desc="Example Creation"):
            examples.extend(text_examples)            
   
        data_file.close()
        
        return examples

def construct_features(inputs):
    example, max_seq_length, tokenizer = inputs
    
    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0   0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambigiously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    if len(input_ids) <= 15:
        return None
    
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)
    masked_lm_labels = list(input_ids)
    vocab_size = len(tokenizer.vocab)
    possible_changes = ['mask', 'random', 'same']
    possible_weights = [0.80, 0.10, 0.10]
    mask_index = tokenizer.vocab['[MASK]']

    # For each token determine appropriate masking.
    for i in range(len(input_ids)):
        if random.random() < 0.15:
            sample_change = random.choices(possible_changes, weights=possible_weights)
            if sample_change == 'mask':
                input_ids[i] = mask_index
            elif sample_change == 'random':
                input_ids[i] = random.randrange(vocab_size)
        else:
            masked_lm_labels[i] = -1

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        masked_lm_labels.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(masked_lm_labels) == max_seq_length

    next_sentence_label = [example.next_sentence_label]

    return InputFeatures(input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids, 
                         masked_lm_labels=masked_lm_labels, next_sentence_label=next_sentence_label)


def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    p = multiprocessing.Pool(multiprocessing.cpu_count())

    for feature in tqdm(p.imap(construct_features, 
                               zip(examples, repeat(max_seq_length), repeat(tokenizer)), chunksize=100), 
                        total=len(examples), desc="Feature Creation"):
        features.append(feature)

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)

def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if param_model.grad is not None:
            if test_nan and torch.isnan(param_model.grad).sum() > 0:
                is_nan = True
            if param_opti.grad is None:
                param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
            param_opti.grad.data.copy_(param_model.grad.data)
        else:
            param_opti.grad = None
    return is_nan

def save_training_dataset(all_input_ids, all_input_mask, all_segment_ids, 
                          all_masked_lm_labels, all_next_sentence_labels, output_dir):
    torch.save(all_input_ids, os.path.join(output_dir, "all_input_ids.pth"))
    torch.save(all_input_mask, os.path.join(output_dir, "all_input_mask.pth"))
    torch.save(all_segment_ids, os.path.join(output_dir, "all_segment_ids.pth"))
    torch.save(all_masked_lm_labels, os.path.join(output_dir, "all_masked_lm_labels.pth"))
    torch.save(all_next_sentence_labels, os.path.join(output_dir, "all_next_sentence_labels.pth"))

def load_training_dataset(output_dir):
    all_input_ids_path = Path(output_dir, "all_input_ids.pth")    

    if all_input_ids_path.exists():
        all_input_ids = torch.load(all_input_ids_path)
        all_input_mask = torch.load(Path(output_dir, "all_input_mask.pth")) 
        all_segment_ids = torch.load(Path(output_dir, "all_segment_ids.pth"))
        all_masked_lm_labels = torch.load(Path(output_dir, "all_masked_lm_labels.pth"))
        all_next_sentence_labels = torch.load(Path(output_dir, "all_next_sentence_labels.pth"))
        return all_input_ids, all_input_mask, all_segment_ids, all_masked_lm_labels, all_next_sentence_labels

    return (None,) * 5

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the data files for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory is where the features and the model checkpoints will be saved.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=512,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=10,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', 
                        type=int, 
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")                       
    parser.add_argument('--optimize_on_cpu',
                        default=False,
                        action='store_true',
                        help="Whether to perform optimization and keep the optimizer averages on CPU")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=128,
                        help='Loss scaling, positive power of 2 values can improve fp16 convergence.')

    args = parser.parse_args()

    processors = {"semeval": SemevalProcessor}

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size // args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    all_input_ids, all_input_mask, all_segment_ids, all_masked_lm_labels, all_next_sentence_labels = load_training_dataset(args.output_dir)

    if all_input_ids is None:
        train_examples = processor.get_train_examples(args.data_dir)
        num_train_steps = int(len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    else:
        num_train_steps = int(len(all_input_ids) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)


    # Prepare model
    model = BertForPreTraining.from_pretrained(args.bert_model, 
                cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(args.local_rank))
    model_path = Path(args.output_dir, "model.pth")

    if model_path.exists():
        model.load_state_dict(torch.load(model_path))

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    if args.fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                            for n, param in model.named_parameters()]
    elif args.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                            for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
        ]
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=t_total)
    
    
    if all_input_ids is None:
        train_features = convert_examples_to_features(train_examples, args.max_seq_length, tokenizer)
        train_features = list(filter(lambda features: features is not None, train_features))

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_masked_lm_labels = torch.tensor([f.masked_lm_labels for f in train_features], dtype=torch.long)
        all_next_sentence_labels = torch.tensor([f.next_sentence_label for f in train_features], dtype=torch.long)
        save_training_dataset(all_input_ids, all_input_mask, all_segment_ids, all_masked_lm_labels, all_next_sentence_labels, args.output_dir)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(all_input_ids))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_masked_lm_labels, all_next_sentence_labels)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    model.train()
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        with tqdm(train_dataloader, desc="Iteration") as pbar:
            for step, batch in enumerate(pbar):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, masked_lm_labels, next_sentence_labels = batch
                loss = model(input_ids, segment_ids, input_mask, masked_lm_labels, next_sentence_labels)
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16 or args.optimize_on_cpu:
                        if args.fp16 and args.loss_scale != 1.0:
                            # scale down gradients for fp16 training
                            for param in model.parameters():
                                if param.grad is not None:
                                    param.grad.data = param.grad.data / args.loss_scale
                        is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                        if is_nan:
                            logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                            args.loss_scale = args.loss_scale / 2
                            model.zero_grad()
                            continue
                        optimizer.step()
                        copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
                    else:
                        optimizer.step()
                    model.zero_grad()
                    pbar.set_postfix(loss=".3f" % loss)

    if n_gpu > 1:
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    main()
