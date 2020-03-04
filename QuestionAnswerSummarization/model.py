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
import time

def siamese(hid1, hid2):
	return torch.exp(-(torch.sum(torch.abs(torch.add(hid1, -hid2)))))
    
def pad_to_length(np_arr, length):
    result = np.zeros(length)
    result[0:np_arr.shape[0]] = np_arr
    return result

def form_input(x):
    return torch.from_numpy(x).float()

def evaluate(device, model_enc, train_exs):
    model_enc.eval()
    #avg_loss = 0.0
    truth_res = []
    pred_res = []
    train_exs0 = train_exs[0]
    train_exs1 = train_exs[1]
    for ex in train_exs0:
        train_vector_one = torch.from_numpy(np.array(ex.indexed_q_one)).to(device)
        train_vector_two = torch.from_numpy(np.array(ex.indexed_q_two)).to(device)
        if len(train_vector_one) == 0 or  len(train_vector_two) == 0:
                pred_res.append(0)
                continue          
        truth_res.append(ex.label)
        label = ex.label
        enc_vec_one = model_enc(train_vector_one)
        enc_vec_two = model_enc(train_vector_two)
                
        distance = siamese(enc_vec_one, enc_vec_two)
        
        pred_res.append(round(distance.item()))

    #print("trainexs0")
    for ex in train_exs1:
        train_vector_one = torch.from_numpy(np.array(ex.indexed_q_one)).to(device)
        train_vector_two = torch.from_numpy(np.array(ex.indexed_q_two)).to(device)
        if len(train_vector_one) == 0 or  len(train_vector_two) == 0:
                pred_res.append(0)
                continue          
        truth_res.append(ex.label)
        label = ex.label
        enc_vec_one = model_enc(train_vector_one)
        enc_vec_two = model_enc(train_vector_two)
                
        distance = siamese(enc_vec_one, enc_vec_two)
        
        pred_res.append(round(distance.item()))
    #print("trainexs1")
    acc = get_accuracy(truth_res, pred_res)
    return acc

class BiLSTMEncoderQuora(nn.Module):

    def __init__(self, device, embedding_dim, hidden_dim, vocab_size):
        super(BiLSTMEncoderQuora, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True).to(device)  # <- change here
        self.device = device
        self.linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.hidden = self.init_hidden()
        self.softmax = nn.Softmax(dim=1)
   
    def init_hidden(self):
        return (autograd.Variable(torch.zeros(2, 1, self.hidden_dim)).to(self.device),   
                autograd.Variable(torch.zeros(2, 1, self.hidden_dim)).to(self.device))    # <- change here: first dim of hidden needs to be doubled

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence).to(self.device)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        y = self.linear(lstm_out[-1])
        return y #self.softmax(y)

class Classifier(nn.Module):
	
	def __init__(self, encoding_dim, hidden_dim_a, hidden_dim_b, output_size):
		super(Classifier, self).__init__()
		self.linear_a = nn.Linear(encoding_dim * 2, hidden_dim_a)
		self.linear_b = nn.Linear(hidden_dim_a, hidden_dim_b)
		self.softmax = nn.Linear(hidden_dim_b, output_size)
		self.softmax_func = nn.Softmax(dim=-1)
	def forward(self, enc_vec):
		out_a = self.linear_a(enc_vec)
		out_b = self.linear_b(out_a)
		return self.softmax_func(self.softmax(out_b))


class BiLSTMEncoder(nn.Module):
    # Parameters: input size (should match embedding layer), hidden size for the LSTM, dropout rate for the RNN
    def __init__(self, input_size, hidden_size, dropout):
        super(BiLSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reduce_h_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.reduce_c_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, dropout=dropout, bidirectional=True)
        self.init_weight()

    # Initializes weight matrices using Xavier initialization
    def init_weight(self):
        # ih: input - hidden
        # hh: hidden - hidden 
        # l0: layer 0
        # l1: layer 1
        # weight_... : weights of input_size x hidden_size or hidden_size x hidden_size (depending on whether ih or hh is specified) are initialized
        # bias_... : bias of size hidden_size initialized
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=1)
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0_reverse, gain=1)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0_reverse, gain=1)
        nn.init.constant_(self.rnn.bias_hh_l0, 0)
        nn.init.constant_(self.rnn.bias_ih_l0, 0)
        nn.init.constant_(self.rnn.bias_hh_l0_reverse, 0)
        nn.init.constant_(self.rnn.bias_ih_l0_reverse, 0)
            
    # embedded_words should be a [batch size x sent len x input dim] tensor
    # input_lens is a tensor containing the length of each input sentence
    # Returns output (each word's representation), context_mask (a mask of 0s and 1s
    # reflecting where the model's output should be considered), and h_t, a *tuple* containing
    # the final states h and c from the encoder for each sentence.
    def forward(self, embedded_words, input_lens):
        # Takes the embedded sentences, "packs" them into an efficient Pytorch-internal representation
        packed_embedding = nn.utils.rnn.pack_padded_sequence(embedded_words, input_lens, batch_first=True)
        # Runs the RNN over each sequence. Returns output at each position as well as the last vectors of the RNN
        # state for each sentence (first/last vectors for bidirectional)
        output, hn = self.rnn(packed_embedding)
        # Unpacks the Pytorch representation into normal tensors
        output, sent_lens = nn.utils.rnn.pad_packed_sequence(output)
        # input_lens is a pytorch Variable which is a wrapping over a tensor that makes it differentiable. 
        # To unpack the underlying tensor, we use input_lens.data.
        max_length = input_lens.data[0].item()
        context_mask = self.sent_lens_to_mask(sent_lens, max_length)

        # Grabs the encoded representations out of hn, which is a weird tuple thing.
        # Note: if you want multiple LSTM layers, you'll need to change this to consult the penultimate layer
        # or gather representations from all layers.

        # output - (sentence_length,batch_size,hid_size*num_directions)
        # hn[0] - (num_layers*num_directions, batch_size, hid_size)
        # hn[1] - (num_layers*num_directions, batch_size, hid_size)

        h, c = hn[0], hn[1]
        # Grab the representations from forward and backward LSTMs
        h_, c_ = torch.cat((h[0], h[1]), dim=1), torch.cat((c[0], c[1]), dim=1)
        # Reduce them by multiplying by a weight matrix so that the hidden size sent to the decoder is the same
        # as the hidden size in the encoder
        new_h = self.reduce_h_W(h_)
        new_c = self.reduce_c_W(c_)
        h_t = (new_h, new_c)
  
        return (output, context_mask, h_t)
      
class LSTMAttnDecoder(nn.Module):
 #we need to take input of max_length of input data for encoder sentences
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_sent_len): #OUTPUT DIM IS THE MAX LENGTH OF INPUT 
        super(LSTMDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.encoder_hidden_output_ll = nn.Linear(hidden_dim*2, hidden_dim, bias = True)
        self.decoder_hidden_state_ll = nn.Linear(hidden_dim, hidden_dim, bias = True)
        self.coverage_ll = nn.Linear(max_sent_len, hidden_dim, bias = True)
        self.get_e_i_t = nn.Linear(hidden_dim, max_sent_len)
        self.get_a_t = nn.Softmax(dim=-1)
        self.first_p_vocab_ll = nn.Linear(hidden_dim*2, hidden_dim, bias = True)
        self.second_p_vocab_ll = nn.Linear(hidden_dim, vocab_size)
        self.get_p_vocab = nn.Softmax(dim=-1)
        self.context_ll = nn.Linear(hidden_dim, 1, bias=True)
        self.s_t_ll = nn.Linear(hidden_dim, 1, bias=True)
        self.x_t_ll = nn.Linear(embedding_dim, 1, bias=True)
        self.sigmoid_for_p_gen = torch.nn.Sigmoid()
        self.init_weight()
              
    def init_weight(self):
        # ih: input - hidden, hh: hidden - hidden, l0: layer 0, l1: layer 1
        # weight_... : weights of input_size x hidden_size or hidden_size x hidden_size (depending on whether ih or hh is specified) are initialized
        # bias_... : bias of size hidden_size initialized
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=1)
        nn.init.constant_(self.rnn.bias_hh_l0, 0)
        nn.init.constant_(self.rnn.bias_ih_l0, 0)
		
    def get_p_gen(context_vector, s_t, embedded):
        return sigmoid_for_p_gen(torch.add(torch.add(self.context_ll(context_vector), self.s_t_ll(s_t)),self.x_t_ll(embedded)))
       
    def forward(self, word, hidden, encoder_outputs, attention_weights, coverage_vec):
        embedded = self.word_embeddings(word) #.view(1, 1, -1) we need this??
        ### useless x = embeds.view(len(word), 1, -1)
        coverage_vec = torch.add(coverage_vec, attention_weights) 
        output, hidden = self.lstm(embedded, hidden)
        s_t = hidden[0][0]
        e_i_t = self.get_e_i_t(torch.tanh(torch.add(self.encoder_hidden_output_ll(encoder_outputs), self.decoder_hidden_state_ll(s_t))))
        
        interim = torch.add(self.encoder_hidden_output_ll(encoder_outputs), self.decoder_hidden_state_ll(s_t))
        e_i_t = self.get_e_i_t(torch.tanh(torch.add(interim, self.coverage_ll(coverage_vec))))
        
        
        
        a_t = self.get_a_t(e_i)
        context_vector = torch.bmm(a_t,encoder_outputs)
        P_vocab = self.get_p_vocab(self.second_p_vocab_ll(self.first_p_vocab_ll(torch.cat((s_t,context_vector), dim=1))))
	
        p_gen = get_p_gen(context_vector, s_t, embedded)
        P_vocab = p_gen*P_vocab + (1-p_gen)*Summ_at
      
        return P_vocab, hidden, a_t, coverage_vec
      
def get_accuracy(truth, pred):
     assert len(truth)==len(pred)
     right = 0
     for i in range(len(truth)):
         if truth[i]==pred[i]:
             right += 1.0
     return right/len(truth)

def train_model(train_exs, word_vectors, device):
    torch.backends.cudnn.benchmark = True    
    train_exs0 = train_exs[0]
    train_exs1 = train_exs[1]
    print(len(train_exs0), len(train_exs1))
    EMBEDDING_DIM = word_vectors.vectors.shape[1]
    HIDDEN_DIM = 100
    OUT_DIM = 1
    EPOCH = 5 
    
    model_enc = BiLSTMEncoderQuora(device, embedding_dim=EMBEDDING_DIM,hidden_dim=HIDDEN_DIM,
                           vocab_size=len(word_vectors.vectors))
    model_enc.to(device)
    #model_sia = Siamese(hid1=HIDDEN_DIM,hid2=HIDDEN_DIM,out=OUT_DIM )
    model_enc.word_embeddings.weight.data = form_input(word_vectors.vectors).to(device)
    optimizer = optim.Adam(model_enc.parameters(),lr = 0.15)
   
    dev_exs = [train_exs0[-15000:], train_exs1[-15000:]]
    best_dev_acc = 0.0001
    no_up = 0
    for epoch in range(EPOCH):
        model_enc.train()
        truth_res = []
        pred_res = []
        avg_loss = 0.0
        count = 0
        perm0 = np.arange(len(train_exs1))[:-30000]
        #perm0 = np.arange(len(train_exs1))[:30]
        random.shuffle(perm0)
        #dev_exs = [train_exs0[31:41], train_exs1[31:41]]
        start_time = time.time()
        print("EPOCH ", epoch)
        for i in perm0:
            #print(i)
            model_enc.train()
            ex0 = train_exs0[i]
            ex1 = train_exs1[i]
            loss = nn.BCELoss()
            loss.to(device)
            for ex in [ex0, ex1]:
                train_vector_one = torch.from_numpy(np.array(ex.indexed_q_one)).long().to(device)
                train_vector_two = torch.from_numpy(np.array(ex.indexed_q_two)).long().to(device)
                if len(train_vector_one) == 0 or  len(train_vector_two) == 0:
                        continue          
                truth_res.append(ex.label)
                label = ex.label
                
                model_enc.hidden = model_enc.init_hidden()
                model_enc.zero_grad()
                
                enc_vec_one = model_enc(train_vector_one)
                enc_vec_two = model_enc(train_vector_two)
                
                distance = siamese(enc_vec_one, enc_vec_two)
                                
                pred_loss = loss(distance, torch.tensor(label).float().to(device))
                
                avg_loss += pred_loss.item()
                pred_loss.backward()
                optimizer.step()
                count = count + 2
	
            if (count%50000==0):
                dev_acc = evaluate(device, model_enc, dev_exs)
                print("Loss: %0.3f, Dev accuracy: %0.3f"% (avg_loss/float(count), dev_acc))

            if (count%10000==0):
                end_time = time.time()
                print("Time taken for 10k iterations: %0.1f"% (end_time - start_time))
                start_time = end_time
    
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            #os.system('rm mr_best_model_acc_*.model')
            print('New Best Dev!!!', best_dev_acc)
            torch.save(model_enc.state_dict(), 'm2r_best_model_acc_' + str(int(dev_acc)) + '.bilstm.model')
            no_up = 0
        else:
            no_up += 1
            if no_up >= 10:
                exit()

    return model_enc

def train_classifier(train_exs,  device):

	EMBEDDING_DIM = 300 #word_vectors.vectors.shape[1]
	HIDDEN_DIM = 100
	hid_a = 80
	hid_b = 50
	out = len(train_exs)
	vocab_size=17615 #len(word_vectors.vectors)

	model_enc = BiLSTMEncoderQuora(device, embedding_dim=EMBEDDING_DIM,hidden_dim=HIDDEN_DIM,vocab_size=vocab_size)
	model_enc.to(device)
	model_enc_pre_train = './mr_best_model_acc_0.bilstm.model'
	model_enc.load_state_dict(torch.load(model_enc_pre_train, map_location={'cuda:0': 'cpu'}))

	classifier = Classifier(encoding_dim=HIDDEN_DIM, hidden_dim_a=hid_a, hidden_dim_b=hid_b, output_size=out)
	classifier.to(device)
	criterion = torch.nn.NLLLoss()
	params = classifier.parameters()
	optimizer = torch.optim.Adam(params, lr=0.001)
	
	best_dev_accuracy = 0.0001

	for epoch in range(10):
		epoch_loss = 0 
		truth_res = []
		pred_res = []
		model_enc.eval()
		classifier.train()
		for ex in train_exs:
			optimizer.zero_grad()
			label = ex.ID
			#print
			input = model_enc(torch.from_numpy(np.array(ex.Q)).long().to(device)).long()
			
			for ques in ex.para_Q_list:
				#print("hii")
				#print(input.size(), enc_ques.size())
				#print(len(ques))
				enc_ques = model_enc(torch.from_numpy(np.array(ques)).long().to(device))
				#print(input.size(), enc_ques.size())
				temp = torch.cat((input, enc_ques.long()), 0)
				input = temp
			#print(input.size())
			#output = model_enc(input)
			mean = input.float().mean(dim=0)
			std = input.float().std(dim=0)
			#print(mean.unsqueeze(0).size())	
			pred = classifier( torch.cat( (mean.unsqueeze(0),std.unsqueeze(0)), dim=1))
			#print("pred",shape)
			#print("label", label)
			#temp = torch.tensor([label]).float()
			#print(temp.shape)
			
			ex_loss =  criterion(pred, torch.LongTensor([label]).to(device))
			epoch_loss +=ex_loss.item() 
			print(epoch, i, pred[0][label], label, ex_loss.item())
			ex_loss.backward()
			optimizer.step()
		#print("epoch_loss", epoch_loss)
		#epoch_loss = 0
	"""
	for ex in test_exs:
		test_vectors = torch.from_numpy(np.array(ex.indexed_words)).long().to(device)
		enc_ques = model_enc(test_vectors)
		pred = classifier(enc_ques)
		pred_label = pred.data.max(1)[1].numpy() torch.tensor(label).float().to(device)
		ex.label = pred_label[0]

	#return test_exs 
	"""
