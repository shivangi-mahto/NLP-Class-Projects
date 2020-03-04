import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable as Var
import torch.autograd as autograd

import numpy as np


# Embedding layer that has a lookup table of symbols that is [full_dict_size x input_dim]. Includes dropout.
# Works for both non-batched and batched inputs
class EmbeddingLayer(nn.Module):
    # Parameters: dimension of the word embeddings, number of words, and the dropout rate to apply
    # (0.2 is often a reasonable value)
    def __init__(self, input_dim, full_dict_size, embedding_dropout_rate):
        super(EmbeddingLayer, self).__init__()
        self.dropout = nn.Dropout(embedding_dropout_rate)
        self.word_embedding = nn.Embedding(full_dict_size, input_dim)

    # Takes either a non-batched input [sent len x input_dim] or a batched input
    # [batch size x sent len x input dim]
    def forward(self, input):
        embedded_words = self.word_embedding(input)
        final_embeddings = self.dropout(embedded_words)
        return final_embeddings


# One-layer RNN encoder for batched inputs -- handles multiple sentences at once. You're free to call it with a
# leading dimension of 1 (batch size 1) but it does expect this dimension.
class RNNEncoder(nn.Module):
    # Parameters: input size (should match embedding layer), hidden size for the LSTM, dropout rate for the RNN,
    # and a boolean flag for whether or not we're using a bidirectional encoder
    def __init__(self, input_size, hidden_size, dropout, bidirect):
        super(RNNEncoder, self).__init__()
        self.bidirect = bidirect
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.reduce_h_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        self.reduce_c_W = nn.Linear(hidden_size * 2, hidden_size, bias=True)
        #self.rnn = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True,
        #                      dropout=dropout, bidirectional=self.bidirect)
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True,
                               dropout=dropout, bidirectional=self.bidirect)
        self.init_weight()

    # Initializes weight matrices using Xavier initialization
    def init_weight(self):
        nn.init.xavier_uniform_(self.rnn.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0, gain=1)
        if self.bidirect:
            nn.init.xavier_uniform_(self.rnn.weight_hh_l0_reverse, gain=1)
            nn.init.xavier_uniform_(self.rnn.weight_ih_l0_reverse, gain=1)
        nn.init.constant_(self.rnn.bias_hh_l0, 0)
        nn.init.constant_(self.rnn.bias_ih_l0, 0)
        if self.bidirect:
            nn.init.constant_(self.rnn.bias_hh_l0_reverse, 0)
            nn.init.constant_(self.rnn.bias_ih_l0_reverse, 0)

    def get_output_size(self):
        return self.hidden_size * 2 if self.bidirect else self.hidden_size

    def sent_lens_to_mask(self, lens, max_length):
        return torch.from_numpy(np.asarray([[1 if j < lens.data[i].item() else 0 for j in range(0, max_length)] for i in range(0, lens.shape[0])]))

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
        max_length = input_lens.data[0].item()
        context_mask = self.sent_lens_to_mask(sent_lens, max_length)

        # Grabs the encoded representations out of hn, which is a weird tuple thing.
        # Note: if you want multiple LSTM layers, you'll need to change this to consult the penultimate layer
        # or gather representations from all layers.
        if self.bidirect:
            h, c = hn[0], hn[1]
            # Grab the representations from forward and backward LSTMs
            h_, c_ = torch.cat((h[0], h[1]), dim=1), torch.cat((c[0], c[1]), dim=1)
            # Reduce them by multiplying by a weight matrix so that the hidden size sent to the decoder is the same
            # as the hidden size in the encoder
            new_h = self.reduce_h_W(h_)
            new_c = self.reduce_c_W(c_)
            h_t = (new_h, new_c)
        else:
            h, c = hn[0][0], hn[1][0]
            h_t = (h, c)
        return (output, context_mask, h_t)

    
    
class LSTMAttnClassifier(nn.Module):
 #args.dim , len(output_indexer), 
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(LSTMAttnClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True)
        
        self.attn = nn.Linear(self.hidden_dim, self.hidden_dim) #*2
        self.Whc = nn.Linear(hidden_dim * 2, hidden_dim) #*2 
        self.Ws = nn.Linear(hidden_dim, vocab_size)
        
        self.hidden2label = nn.Linear(hidden_dim, vocab_size)
        self.hidden = self.init_hidden()
        self.softmax = nn.LogSoftmax(dim=1)
        self.init_weight()
    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))
    def init_weight(self):
        nn.init.xavier_uniform_(self.lstm.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0, gain=1)
        nn.init.constant_(self.lstm.bias_hh_l0, 0)
        nn.init.constant_(self.lstm.bias_ih_l0, 0)
    def forward(self, word, hidden, encoder_output):

        lstm_out, self.hidden = self.lstm(word, hidden)

        encoder_output = encoder_output.view(-1, self.hidden_dim ) #* 2)
        
        attn_h = self.attn(self.hidden[0])
        attn_prod = torch.mm(attn_h[0], encoder_output.t())
        attn_weights = F.softmax(attn_prod, dim=1)
        context = torch.mm(attn_weights, encoder_output)
        hc = torch.cat([self.hidden[0].view(-1, self.hidden_dim), context], dim=1)
        
        out_hc = torch.tanh(self.Whc(hc))
        y = self.softmax(self.Ws(out_hc))
        return y.unsqueeze(0) , self.hidden

class LSTMCovClassifier(nn.Module):
 #args.dim , len(output_indexer), 
    def __init__(self, embedding_dim, hidden_dim, vocab_size, max_sent_len):
        super(LSTMCovClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True)
        
        self.attn = nn.Linear(self.hidden_dim, self.hidden_dim ) #*2)
        self.Whc = nn.Linear(hidden_dim * 2, hidden_dim) #3
        self.Ws = nn.Linear(hidden_dim, vocab_size)
        self.Cov_ll = nn.Linear(max_sent_len, self.hidden_dim) #*2)

        
        self.hidden2label = nn.Linear(hidden_dim, vocab_size)
        self.hidden = self.init_hidden()
        self.softmax = nn.LogSoftmax(dim=1)
        self.init_weight()
    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))
    def init_weight(self):
        nn.init.xavier_uniform_(self.lstm.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0, gain=1)
        nn.init.constant_(self.lstm.bias_hh_l0, 0)
        nn.init.constant_(self.lstm.bias_ih_l0, 0)
    def forward(self, word, hidden, encoder_output, coverage_vec, attention_weights):

        coverage_vec = torch.add(coverage_vec, attention_weights) 
        
        lstm_out, self.hidden = self.lstm(word, hidden)
        encoder_output = encoder_output.view(-1, self.hidden_dim) # * 2)
        
        attn_h = self.attn(self.hidden[0])
        
        #print()
        con_ll = self.Cov_ll(coverage_vec)
        #print(con_ll.size())
        attn_cov = torch.add(attn_h,con_ll)
        #print("adding cov",attn_cov.size())
        #print("attn with h", attn_h.size())
        attn_prod = torch.mm(attn_cov[0], encoder_output.t())
        #attn_prod = torch.mm(attn_h[0], encoder_output.t())
        attn_weights = F.softmax(attn_prod, dim=1)
        context = torch.mm(attn_weights, encoder_output)
        hc = torch.cat([self.hidden[0].view(-1, self.hidden_dim), context], dim=1)
        
        out_hc = torch.tanh(self.Whc(hc))
        y = self.softmax(self.Ws(out_hc))
        #print("attnszie",attn_weights.size())
        return y.unsqueeze(0) , self.hidden, coverage_vec, attn_weights


class LSTMQuesCovClassifier(nn.Module):
 #args.dim , len(output_indexer), 
    def __init__(self, embedding_dim, ques_dim, hidden_dim, vocab_size, max_sent_len):
        super(LSTMQuesCovClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_dim+ques_dim, hidden_dim, num_layers=1, batch_first=True)
        
        self.attn = nn.Linear(self.hidden_dim, self.hidden_dim ) #*2)
        self.Whc = nn.Linear(hidden_dim * 2, hidden_dim) #3
        self.Ws = nn.Linear(hidden_dim, vocab_size)
        self.Cov_ll = nn.Linear(max_sent_len, self.hidden_dim) #*2)

        
        self.hidden2label = nn.Linear(hidden_dim, vocab_size)
        self.hidden = self.init_hidden()
        self.softmax = nn.LogSoftmax(dim=1)
        self.init_weight()
    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))
    def init_weight(self):
        nn.init.xavier_uniform_(self.lstm.weight_hh_l0, gain=1)
        nn.init.xavier_uniform_(self.lstm.weight_ih_l0, gain=1)
        nn.init.constant_(self.lstm.bias_hh_l0, 0)
        nn.init.constant_(self.lstm.bias_ih_l0, 0)
    def forward(self, word, hidden, encoder_output, coverage_vec, attention_weights):

        coverage_vec = torch.add(coverage_vec, attention_weights) 
        
        lstm_out, self.hidden = self.lstm(word, hidden)
        encoder_output = encoder_output.view(-1, self.hidden_dim) # * 2)
        
        attn_h = self.attn(self.hidden[0])
        
        con_ll = self.Cov_ll(coverage_vec)
        #print(con_ll.size())
        attn_cov = torch.add(attn_h,con_ll)
        #print("adding cov",attn_cov.size())
        #print("attn with h", attn_h.size())
        attn_prod = torch.mm(attn_cov[0], encoder_output.t())
        #attn_prod = torch.mm(attn_h[0], encoder_output.t())
        attn_weights = F.softmax(attn_prod, dim=1)
        context = torch.mm(attn_weights, encoder_output)
        hc = torch.cat([self.hidden[0].view(-1, self.hidden_dim), context], dim=1)
        
        out_hc = torch.tanh(self.Whc(hc))
        y = self.softmax(self.Ws(out_hc))
        #print("attnszie",attn_weights.size())
        return y.unsqueeze(0) , self.hidden, coverage_vec, attn_weights

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
        x = embeds.view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y = self.linear(lstm_out[-1])
        return y #self.softmax(y)