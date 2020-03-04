# models.py

from sentiment_data import *
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

#Expected results
#Intialized with pretrained word embedding as intialization and 1 element peak everytime
#LSTM 1 hidden/ 1 context/ 0.8189
# expected results BiLSTM  1 element peak 0.8208


#hidden , context - 
class FFNN(nn.Module):
    def __init__(self, inp, hid1, hid2, out): # OUT IS JUST 1 DIM, INP IS EITHER 50 OR 300, HID1 AND HID2 DECIDE BY YOURSELF
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(inp, hid1)
        self.fc2 = nn.Linear(hid1, hid2)
        self.fc3 = nn.Linear(hid2, out)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        # Initialize weights according to the Xavier Glorot formula
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        #
    # Forward computation. Backward computation is done implicitly (nn.Module already has an implementation of
    # it that you shouldn't need to override)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)#nn.Softmax(dim=0)

# Returns a new numpy array with the data from np_arr padded to be of length length. If length is less than the
# length of the base array, truncates instead.
def pad_to_length(np_arr, length):
    result = np.zeros(length)
    result[0:np_arr.shape[0]] = np_arr
    return result

def form_input(x):
    return torch.from_numpy(x).float()

def ffnn_evaluate(NET, seq_avg_emb, dl2d):
    probs = NET.forward(form_input(seq_avg_emb))
    sum_all = 0
    
    for i in range(0,seq_avg_emb.shape[0]):    
        prediction = torch.argmax(probs[i])
        if(prediction.numpy()==dl2d[i]):
            sum_all = sum_all + 1
    return sum_all
# Train a feedforward neural network on the given training examples, using dev_exs for development and returning
# predictions on the *blind* test_exs (all test_exs have label 0 as a dummy placeholder value). Returned predictions
# should be SentimentExample objects with predicted labels and the same sentences as input (but these won't be
# read for evaluation anyway)
def train_ffnn(train_exs, dev_exs, test_exs, word_vectors):
    # 59 is the max sentence length in the corpus, so let's set this to 60
    seq_max_len = 60
    Vector_dict = word_vectors.vectors
    len_word_emb = Vector_dict.shape[1]   

    # Labels
    #train_labels = np.array([ex.label for ex in train_exs])
    # To get you started off, we'll pad the training input to 60 words to make it a square matrix.
    train_seq_lens = np.array([len(ex.indexed_words) for ex in train_exs])
    dev_seq_lens = np.array([len(ex.indexed_words) for ex in dev_exs])

    #READ TRAIN DATA
    train_vectors = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in train_exs])
    train_labels = (([ex.label for ex in train_exs]))
    tl2d = np.zeros([len(train_labels),1])
    tl2d[:,-1] = train_labels
    
    len_train_data = train_vectors.shape[0]
    seq_avg_emb = np.zeros([len_train_data, len_word_emb]) # AVG OVER WORD EMBEDDING
    for i in range(0,len_train_data):
        for j in range(0, train_seq_lens[i]):
            seq_avg_emb[i] = seq_avg_emb[i] + Vector_dict[int(train_vectors[i][j])]
        seq_avg_emb[i] = seq_avg_emb[i]/float(train_seq_lens[i])
    
    dev_vectors = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len) for ex in dev_exs])
    dev_labels = (([ex.label for ex in dev_exs]))
    dl2d = np.zeros([len(dev_labels),1]) #,1
    dl2d[:,-1] = dev_labels
    len_dev_data = dev_vectors.shape[0]
    seq_avg_emb_dev = np.zeros([len_dev_data, len_word_emb])
    for i in range(0,len_dev_data):
        for j in range(0, dev_seq_lens[i]):
            seq_avg_emb_dev[i] = seq_avg_emb_dev[i] + Vector_dict[int(dev_vectors[i][j])]
        seq_avg_emb_dev[i] = seq_avg_emb_dev[i]/dev_seq_lens[i]#float(seq_max_len)

    batch_size = 25
    num_classes = 2
    ffnn = FFNN(len_word_emb,150,50,num_classes)
    lr = 0.0001
    optimizer = optim.Adam(ffnn.parameters(), lr)
    y_onehot = torch.FloatTensor(batch_size, num_classes)
    number_of_batches = math.floor(len_train_data/batch_size)
    #init_weights = copy.deepcopy(ffnn.fc1.weight.data)
    for epoch in range(40): #40
        #arr = np.arange(len_train_data)
        #np.random.shuffle(arr)
    
        _seq_avg_emb = seq_avg_emb#[arr,:]
        _seq_labels  = tl2d#[arr,:]
        total_loss = 0.0
        
        for i in range(0,number_of_batches):
            
            train_data_batch = form_input(_seq_avg_emb[batch_size*i:batch_size*(i+1),:])
            train_labels_batch = _seq_labels[batch_size*i:batch_size*(i+1),:] #,:

            x = train_data_batch
            y = train_labels_batch
            y_onehot.zero_()
            y_onehot.scatter_(1, torch.from_numpy(np.asarray(y,dtype=np.long)), 1)

            ffnn.zero_grad()
            probs = ffnn.forward(x)
            
            loss = torch.sum(torch.neg(torch.log(probs))* y_onehot)/batch_size
            total_loss += loss
            loss.backward()
            optimizer.step()        
           
        sum1 = ffnn_evaluate(ffnn, seq_avg_emb, tl2d)
        sum2 = ffnn_evaluate(ffnn, seq_avg_emb_dev, dl2d) 
    
        print("Loss per epoch %i: %f and acc. train: %0.3f , dev : %0.3f "% (epoch, total_loss/number_of_batches, sum1/len_train_data, sum2/len_dev_data))
    

    for ex in test_exs:
        test_vectors = np.asarray([pad_to_length(np.array(ex.indexed_words), seq_max_len)])
        
        seq_avg_emb_test = np.zeros([1,len_word_emb])
        for j in range(0, len(ex.indexed_words)):
            seq_avg_emb_test = seq_avg_emb_test + Vector_dict[int(test_vectors[0][j])]
        seq_avg_emb_test = seq_avg_emb_test/float(len(ex.indexed_words))
        
        probs = ffnn.forward(form_input(seq_avg_emb_test))
        pred_label = probs.data.max(1)[1].numpy() 
        #pred_res.append(pred_label)
        ex.label = pred_label[0]
        
        
    return test_exs

def evaluate(model, train_exs):
    model.eval()
    #avg_loss = 0.0
    truth_res = []
    pred_res = []
    
    for ex in train_exs:
        train_vectors = torch.from_numpy(np.array(ex.indexed_words))
        truth_res.append(ex.label)
        model.hidden = model.init_hidden()
        #model.zero_grad()
        pred = model(train_vectors)
        pred_label = pred.data.max(1)[1].numpy() 
        pred_res.append(pred_label)

    acc = get_accuracy(truth_res, pred_res)
    return acc

class MultiLSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size,dropout):
        super(MultiLSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, dropout=dropout)
        #self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()
       
        self.softmax = nn.Softmax(dim=1)
        
    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))
                
 
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y  = self.hidden2label(lstm_out[-1])
        #log_probs = F.log_softmax(y)
        return self.softmax(y)
    
class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        #self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()
        self.softmax = nn.Softmax(dim=1)
        
    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y  = self.hidden2label(lstm_out[-1])
        #log_probs = F.log_softmax(y)
        return self.softmax(y)

class BiLSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size):
        super(BiLSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)  # <- change here

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim * 2, label_size)
        self.hidden = self.init_hidden()
        self.softmax = nn.Softmax(dim=1)
   
    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers * num_directions, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(2, 1, self.hidden_dim)),   
                autograd.Variable(torch.zeros(2, 1, self.hidden_dim)))    # <- change here: first dim of hidden needs to be doubled

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        y = self.hidden2tag(lstm_out[-1])
        #tag_scores = F.log_softmax(tag_space, dim=1)
        return self.softmax(y)
    
def get_accuracy(truth, pred):
     assert len(truth)==len(pred)
     right = 0
     for i in range(len(truth)):
         if truth[i]==pred[i]:
             right += 1.0
     return right/len(truth)

def train_fancy(train_exs, dev_exs, test_exs, word_vectors):
    
    Vector_dict = word_vectors.vectors
    len_word_emb = Vector_dict.shape[1]   

    EMBEDDING_DIM = len_word_emb
    HIDDEN_DIM = 100
    EPOCH = 1 # 1 is best for case 1
    num_classes =2

    
    model = BiLSTMClassifier(embedding_dim=EMBEDDING_DIM,hidden_dim=HIDDEN_DIM,
                           vocab_size=len(Vector_dict),label_size=num_classes)
    
    model.word_embeddings.weight.data = form_input(Vector_dict)
    optimizer = optim.Adam(model.parameters(),lr = 1e-3)
    y_onehot = torch.FloatTensor(1, num_classes)
    tl2d = np.zeros([1,1])
    for i in range(EPOCH):
        #random.shuffle(train_data)
        model.train()
        truth_res = []
        pred_res = []
        avg_loss = 0.0
        count = 0
        best_dev_acc = 0.0001
        no_up = 0
        #for ex in train_exs:
        for i in range(0,len(train_exs)):
            ex = train_exs[i]
            train_vectors = torch.from_numpy(np.array(ex.indexed_words))
            truth_res.append(ex.label)
            tl2d[:,-1] = ex.label
            
            y_onehot.zero_()
            y_onehot.scatter_(1, torch.from_numpy(np.asarray(tl2d,dtype=np.long)), 1)

            model.hidden = model.init_hidden()
            model.zero_grad()
            pred = model(train_vectors)
            pred_label = pred.data.max(1)[1].numpy() 
            pred_res.append(pred_label)
            #print(pred_label[0])
            loss = torch.sum(torch.neg(torch.log(pred))* y_onehot)

            avg_loss += loss
            loss.backward()
            optimizer.step()
            
            #print (get_accuracy(truth_res,pred_res))
            count = count + 1
            
            if (count%500==0):
                train_acc = evaluate(model, train_exs)
                dev_acc =evaluate(model, dev_exs)
                print(pred)
                print("Loss per itr %i: %0.3f and accuracy on training data: %0.3f and dev data: %0.3f"% (count, avg_loss/float(count), train_acc, dev_acc))
        dev_acc =evaluate(model, dev_exs)
        print("final dev_acc:0.3f",dev_acc)
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            #os.system('rm mr_best_model_acc_*.model')
            print('New Best Dev!!!')
            torch.save(model.state_dict(), 'mr_best_model_acc_' + str(int(dev_acc)) + '.bilstm.model')
            no_up = 0
        else:
            no_up += 1
            if no_up >= 10:
                exit()

    for ex in test_exs:
        #dev_acc = 0
        #model.load_state_dict(model.state_dict(), 'mr_best_model_acc_' + str(int(dev_acc)) + '.model')
        test_vectors = torch.from_numpy(np.array(ex.indexed_words))
        model.hidden = model.init_hidden()
        #model.zero_grad()
        pred = model(test_vectors)
        pred_label = pred.data.max(1)[1].numpy() 
        #pred_res.append(pred_label)
        ex.label = pred_label[0]
        
        #probs = ffnn.forward(form_input(seq_avg_emb_test))
        #prediction = torch.argmax(probs)
        #ex.label= prediction.numpy()
        
    return test_exs


