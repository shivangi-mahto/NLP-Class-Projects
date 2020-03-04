#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 18:28:28 2018

@author: shivi
"""
import argparse
import random
import numpy as np
import time
import torch
from torch import optim
#from lf_evaluator import *
from model_aquila import EmbeddingLayer, LSTMAttnDecoder, BiLSTMEncoder
#from data import *
import torch.nn.functional as F
from utils import *

PAD_SYMBOL = "<PAD>"
SOS_SYMBOL = "<SOS>"
EOS_SYMBOL = "<EOS>"

def encode_input_for_decoder(x_tensor, inp_lens_tensor, model_input_emb, model_enc):
    input_emb = model_input_emb.forward(x_tensor)
    (enc_output_each_word, enc_final_states) = model_enc.forward(input_emb, inp_lens_tensor)
    enc_final_states_reshaped = (enc_final_states[0].unsqueeze(0), enc_final_states[1].unsqueeze(0))
    return (enc_output_each_word, enc_final_states_reshaped)


def train_model_encdec(train_data, test_data, indexer, args, device):
    # Sort in descending order by x_indexed, essential for pack_padded_sequence
    train_data.sort(key=lambda ex: len(ex.doc), reverse=True)
    test_data.sort(key=lambda ex: len(ex.doc), reverse=True)

    print("start", len(train_data))
    # Create indexed input
    input_max_len = np.max(np.asarray([len(ex.doc) for ex in train_data]))
    all_train_input_data = make_padded_input_tensor(train_data, indexer, input_max_len, args.reverse_input)
    all_test_input_data = make_padded_input_tensor(test_data, indexer, input_max_len, args.reverse_input)

    output_max_len = np.max(np.asarray([len(ex.best_x) for ex in train_data]))
    all_train_output_data = make_padded_output_tensor(train_data, indexer, output_max_len)
    all_test_output_data = make_padded_output_tensor(test_data, indexer, output_max_len)

    #print("Train length: %i" % input_max_len)
   # print("Train output length: %i" % np.max(np.asarray([len(ex.best_x) for ex in train_data])))
    #print("Train matrix: %s; shape = %s" % (all_train_input_data, all_train_input_data.shape))
    #print("x and y", train_data[0].doc, train_data[0].best_x)
    
    # Create model
    #print(args.input_dim, len(indexer), args.emb_dropout)
        
    model_enc = BiLSTMEncoder(300, 200, 0.2) #, args.bidirectional)
    model_enc.to(device)
    print(args.output_dim, len(indexer), args.emb_dropout )
    
    #model_dec = RNNDecoder(args.output_dim, args.hidden_size, len(output_indexer), dropout=0.15)
    model_dec = LSTMAttnDecoder(300, 200, 17618,input_max_len, dropout=0.15)
    model_dec.to(device)
    print("stage 1")
    model_input_emb = EmbeddingLayer(300, 17618, 0.2)
    model_input_emb.to(device)
    model_output_emb = EmbeddingLayer(300, 17618, 0.2)
    model_output_emb.to(device)
    #model_dec = AttnDecoderRNN(args.output_dim, args.hidden_size, len(output_indexer), dropout_p=0.1, input_max_len)
    
    sos_ind = indexer.index_of(SOS_SYMBOL)
    eos_idx = indexer.index_of(EOS_SYMBOL)
    word_lens_tensor = torch.tensor(1).unsqueeze(0)
  
    #criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.NLLLoss()
    params = list(model_enc.parameters()) + list(model_dec.parameters()) + list(model_input_emb.parameters()) + list(model_output_emb.parameters()) #list(model_dec.parameters()) +
    optimizer = torch.optim.Adam(params, lr=0.001)
    
    for epoch in range(0,1):  #try for 10
        model_enc.train()
        model_dec.train()
        model_input_emb.train()
        model_output_emb.train()
        
        perm0 =np.arange(len(train_data))
        #random.shuffle(perm0)
        for idx in perm0:
            
            model_enc.zero_grad()
            model_dec.zero_grad()
            model_input_emb.zero_grad()
            model_output_emb.zero_grad()
            sent_loss = torch.autograd.Variable(torch.FloatTensor([0]))
             
            #encoding input
            x_tensor = torch.from_numpy(all_train_input_data[idx]).unsqueeze(0)
            inp_lens_tensor = torch.from_numpy(np.array(len(train_data[idx].doc))).unsqueeze(0)
            enc_word, context = encode_input_for_decoder(x_tensor, inp_lens_tensor, model_input_emb, model_enc)
            first_idx_tensor = torch.tensor(sos_ind).unsqueeze(0)
            sos_embed = model_output_emb.forward(first_idx_tensor).unsqueeze(0)
            hidden = context
            
            #pred, hidden = model_dec.forward(sos_embed, word_lens_tensor,hidden)
            attention_weights =  torch.autograd.Variable(torch.FloatTensor([0]*len(train_data[idx].doc)))
            coverage_vec = torch.autograd.Variable(torch.FloatTensor([0]*len(train_data[idx].doc)))
            pred, hidden, a_t, coverage = model_dec.forward(sos_embed, word_lens_tensor,hidden,enc_word, attention_weights, coverage_vec)
            hidden = (hidden[0].unsqueeze(0), hidden[1].unsqueeze(0))
            #word_idx_tensor = first_idx_tensor  embedded, indicator_vec, hidden, encoder_outputs, attention_weights, coverage_vec
            #print("input_size",len(context),inp_lens_tensor)
            #first to last output words as input
            
            for tgt_idx in train_data[idx].best_x:
                
                sent_loss += criterion(pred, torch.LongTensor([tgt_idx]))
                if tgt_idx == eos_idx:
                    break
                word_emb = model_output_emb.forward(torch.tensor(tgt_idx).unsqueeze(0)).unsqueeze(0)
                
                pred, hidden, a_t, coverage = model_dec.forward(word_emb, word_lens_tensor,hidden,enc_word,a_t, coverage)
                #pred, hidden = model_dec.forward(word_emb, word_lens_tensor,hidden)
                #print(pred)
                hidden = (hidden[0].unsqueeze(0), hidden[1].unsqueeze(0))
                #tgt_idx_tensor = torch.LongTensor(tgt_idx).unsqueeze(0)
                
                #print(max(pred))
                #word_idx_tensor = tgt_idx_tensor
                
                #print(sent_loss, word_loss.item())
           
            
            #model_enc.init_weight()
            print(sent_loss)
            sent_loss.backward()
            optimizer.step()
            
    # Loop over epochs, loop over examples, given some indexed words, call encode_input_for_decoder, then call your
    # decoder, accumulate losses, update parameters
    #print("finished")
    #raise Exception("Implement the rest of me to train your parser")
    return Seq2SeqSemanticParser(model_enc,model_dec,model_input_emb,model_output_emb,input_indexer,output_indexer,args.reverse_input )

def make_padded_input_tensor(exs, input_indexer, max_len, reverse_input):
    if reverse_input:
        return np.array(
            [[ex.doc[len(ex.doc) - 1 - i] if i < len(ex.doc) else input_indexer.index_of(PAD_SYMBOL)
              for i in range(0, max_len)]
             for ex in exs])
    else:
        return np.array([[ex.doc[i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
                          for i in range(0, max_len)]
                         for ex in exs])

# Analogous to make_padded_input_tensor, but without the option to reverse input
def make_padded_output_tensor(exs, output_indexer, max_len):
    return np.array([[ex.best_x[i] if i < len(ex.best_x) else output_indexer.index_of(PAD_SYMBOL) for i in range(0, max_len)] for ex in exs])

def len_input(exs):
    return np.array([[len(ex.doc)] for ex in exs])
