#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 23:13:17 2018

@author: shivi
"""


from data import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import random 
import math
import copy
import os
import torch.autograd as autograd

def siamese(hid1, hid2):
	return torch.exp(-(torch.sum(torch.abs(torch.add(hid1, -hid2)))))
    
def pad_to_length(np_arr, length):
    result = np.zeros(length)
    result[0:np_arr.shape[0]] = np_arr
    return result

def form_input(x):
    return torch.from_numpy(x).float()

def evaluate(model_enc, train_exs):
    model_enc.eval()
    #avg_loss = 0.0
    truth_res = []
    pred_res = []
    train_exs0 = train_exs[0]
    train_exs1 = train_exs[1]
    for ex in train_exs0[0:2000]:
        train_vector_one = torch.from_numpy(np.array(ex.indexed_q_one))
        train_vector_two = torch.from_numpy(np.array(ex.indexed_q_two))
        
        truth_res.append(ex.label)
        label = ex.label
        enc_vec_one = model_enc(train_vector_one)
        enc_vec_two = model_enc(train_vector_two)
                
        distance = siamese(enc_vec_one, enc_vec_two)
        
        pred_res.append(round(distance.item()))

    print("trainexs0")
    for ex in train_exs1[0:2000]:
        train_vector_one = torch.from_numpy(np.array(ex.indexed_q_one))
        train_vector_two = torch.from_numpy(np.array(ex.indexed_q_two))
        
        truth_res.append(ex.label)
        label = ex.label
        enc_vec_one = model_enc(train_vector_one)
        enc_vec_two = model_enc(train_vector_two)
                
        distance = siamese(enc_vec_one, enc_vec_two)
        
        pred_res.append(round(distance.item()))
    print("trainexs1")
    acc = get_accuracy(truth_res, pred_res)
    return acc

class BiLSTMEncoder(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(BiLSTMEncoder, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)  # <- change here

        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.hidden = self.init_hidden()
        self.softmax = nn.Softmax(dim=1)
   
    def init_hidden(self):
        return (autograd.Variable(torch.zeros(2, 1, self.hidden_dim)),   
                autograd.Variable(torch.zeros(2, 1, self.hidden_dim)))    # <- change here: first dim of hidden needs to be doubled

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        y = self.linear(lstm_out[-1])
        return y #self.softmax(y)
    
def get_accuracy(truth, pred):
     assert len(truth)==len(pred)
     right = 0
     for i in range(len(truth)):
         if truth[i]==pred[i]:
             right += 1.0
     return right/len(truth)

def train_model(train_exs, word_vectors):
    
    Vector_dict = word_vectors.vectors
    len_word_emb = Vector_dict.shape[1]   

    train_exs0 = train_exs[0]
    train_exs1 = train_exs[1]
    print(len( train_exs0), len( train_exs1))
    EMBEDDING_DIM = len_word_emb
    HIDDEN_DIM = 100
    OUT_DIM = 1
    EPOCH = 1 
    
    model_enc = BiLSTMEncoder(embedding_dim=EMBEDDING_DIM,hidden_dim=HIDDEN_DIM,
                           vocab_size=len(Vector_dict))
    #model_sia = Siamese(hid1=HIDDEN_DIM,hid2=HIDDEN_DIM,out=OUT_DIM )
    model_enc.word_embeddings.weight.data = form_input(Vector_dict)
    optimizer = optim.Adam(model_enc.parameters(),lr = 1e-3)
   
    for i in range(EPOCH):
        #random.shuffle(train_data)
        model_enc.train()
        #model_sia.train()
        truth_res = []
        pred_res = []
        avg_loss = 0.0
        count = 0
        best_dev_acc = 0.0001
        no_up = 0
        #for ex in train_exs:
        loss = nn.BCELoss()
        perm0 = np.arange(len(train_exs1))
        #perm0 = np.arange(10000)
        random.shuffle(perm0)

        for i in perm0:
            #print(i)
            ex0 = train_exs0[i]
            ex1 = train_exs1[i]
            
            for ex in [ex0, ex1]:
                train_vector_one = torch.from_numpy(np.array(ex.indexed_q_one)).long()
                train_vector_two = torch.from_numpy(np.array(ex.indexed_q_two)).long()
                
                truth_res.append(ex.label)
                label = ex.label
                
                model_enc.hidden = model_enc.init_hidden()
                model_enc.zero_grad()
                
                enc_vec_one = model_enc(train_vector_one)
                enc_vec_two = model_enc(train_vector_two)
                
                distance = siamese(enc_vec_one, enc_vec_two)
                				#pred_res.append(distance.item())
                                
                pred_loss = loss(distance, torch.tensor(label).float())
                
                avg_loss += pred_loss.item()
                pred_loss.backward()
                optimizer.step()
            count = count + 2
    
            if (count%1000==0):
                train_acc = evaluate(model_enc, train_exs)
                print("Loss per itr %i: %0.3f and accuracy on training data: %0.3f"% (count, avg_loss/float(count), train_acc))

                #dev_acc   = evaluate(model, dev_exs)
                #print("Loss per itr %i: %0.3f and accuracy on training data: %0.3f and dev data: %0.3f"% (count, avg_loss/float(count), train_acc, dev_acc))
        #dev_acc =evaluate(model, dev_exs)
        #print("final dev_acc:0.3f",dev_acc)
    """
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            #os.system('rm mr_best_model_acc_*.model')
            print('New Best Dev!!!')
            torch.save(model_enc.state_dict(), 'mr_best_model_acc_' + str(int(dev_acc)) + '.bilstm.model')
            no_up = 0
        else:
            no_up += 1
            if no_up >= 10:
                exit()

    for ex in test_exs:
        test_vectors = torch.from_numpy(np.array(ex.indexed_words))
        model_enc.hidden = model_enc.init_hidden()
        pred = model_enc(test_vectors)
        pred_label = pred.data.max(1)[1].numpy() 
        ex.label = pred_label[0]

    return test_exs
	"""
