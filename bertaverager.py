# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team and Team MVP.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.

from pytorch_pretrained_bert.modeling import *
import torch
from torch import nn
from torch.nn import NLLLoss

class BertForSplicedSequenceClassification(BertForSequenceClassification):
    """BERT model for classification which averages over slices of its input.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output."""
    def __init__(self, config, num_labels=2, num_splices=None):
        super(BertForSplicedSequenceClassification, self).__init__(config=config, num_labels=num_labels)
        self.num_splices = num_splices
        self.softmax = nn.Softmax(dim=1)
        self.loc_softmax = nn.Softmax(dim=0)
        #self.loc_weights = nn.Parameter(torch.ones(num_splices))
        self.loss_fct = NLLLoss()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        #print('Shape of input_ids', input_ids.size())
        spliced_input_ids = torch.chunk(input_ids, self.num_splices, dim=1)
        spliced_token_type_ids = torch.chunk(token_type_ids, self.num_splices, dim=1)
        spliced_attention_mask = torch.chunk(attention_mask, self.num_splices, dim=1)
        
        #print('spliced input ids', spliced_input_ids)
        #print('spliced token type ids', spliced_token_type_ids)
        #print('spliced attn mask', spliced_attention_mask)

        pooled_outputs = map(lambda x: self.bert(x[0], x[1], x[2], output_all_encoded_layers=False)[1], zip(spliced_input_ids, spliced_token_type_ids, spliced_attention_mask))
        pooled_outputs = map(lambda x: self.dropout(x), pooled_outputs)
        logits = map(lambda x: self.classifier(x), pooled_outputs)
        
        #print('logits', logits)
        class_probabilities = map(lambda x: self.softmax(x), logits)
        #print('class probabilities', class_probabilities)

        #loc_probabilities = self.loc_softmax(self.loc_weights)
        #print('loc probabilities', loc_probabilities)
        overall_probabilities = 0.0
                
        for class_probability in class_probabilities:
            overall_probabilities += class_probability / self.num_splices

        #print('overall probabilities', overall_probabilities)
        #print('labels', labels)
        
        if labels is not None:
            loss = self.loss_fct(overall_probabilities, labels)
            #print('loss', loss)
            return loss
        else:
            return overall_probabilities

