import argparse
import random
import numpy as np
import time
import torch
from torch import optim
from lf_evaluator import *
from models import *
from data import *
import torch.nn.functional as F
from utils import *

#Expected result after 30 epochs on AttnRNNDecoder with forward hidden layer of encoder as encoding states
#Exact logical form matches: 61 / 120 = 0.508
#Token-level accuracy: 3183 / 3908 = 0.814
#Denotation matches: 66 / 120 = 0.550
#print (SOS_SYMBOL)
#print (EOS_SYMBOL)

#for20
#Exact logical form matches: 68 / 120 = 0.567
#Token-level accuracy: 3182 / 3908 = 0.814
#Denotation matches: 73 / 120 = 0.608
# 0.8 20 epochs
#Exact logical form matches: 59 / 120 = 0.492
#Token-level accuracy: 3102 / 3908 = 0.794
#Denotation matches: 65 / 120 = 0.542
#0.8 for 40 epcohs
#Exact logical form matches: 65 / 120 = 0.542
#Token-level accuracy: 3112 / 3908 = 0.796
#Denotation matches: 68 / 120 = 0.567
#0.5
#Exact logical form matches: 50 / 120 = 0.417
##Token-level accuracy: 3187 / 3908 = 0.816
#Denotation matches: 54 / 120 = 0.450
#beam search 1 beam - 30 epcohs
#Exact logical form matches: 65 / 120 = 0.542
#Token-level accuracy: 3094 / 3908 = 0.792
#Denotation matches: 68 / 120 = 0.567
#beam size 3 - 30 epochs
#Exact logical form matches: 61 / 120 = 0.508
#Token-level accuracy: 3076 / 3908 = 0.787
#Denotation matches: 66 / 120 = 0.550
# Semantic parser that uses Jaccard similarity to find the most similar input example to a particular question and
# returns the associated logical form.
#30 epoch 3 beam size
#Exact logical form matches: 68 / 120 = 0.567
#Token-level accuracy: 3151 / 3908 = 0.806
#Denotation matches: 72 / 120 = 0.600
#beam size - 3 50 epochs
#Exact logical form matches: 68 / 120 = 0.567
#Token-level accuracy: 3178 / 3908 = 0.813
#Denotation matches: 74 / 120 = 0.617
#epoch 10 3 beam size
#Exact logical form matches: 53 / 120 = 0.442
#Token-level accuracy: 3075 / 3908 = 0.787
#Denotation matches: 67 / 120 = 0.558
def _parse_args():
    parser = argparse.ArgumentParser(description='main.py')
    
    # General system running and configuration options
    parser.add_argument('--do_nearest_neighbor', dest='do_nearest_neighbor', default=False, action='store_true', help='run the nearest neighbor model')

    parser.add_argument('--train_path', type=str, default='data/geo_train.tsv', help='path to train data')
    parser.add_argument('--dev_path', type=str, default='data/geo_dev.tsv', help='path to dev data')
    parser.add_argument('--test_path', type=str, default='data/geo_test.tsv', help='path to blind test data')
    parser.add_argument('--test_output_path', type=str, default='geo_test_output.tsv', help='path to write blind test results')
    parser.add_argument('--domain', type=str, default='geo', help='domain (geo for geoquery)')
    
    # Some common arguments for your convenience
    parser.add_argument('--seed', type=int, default=0, help='RNG seed (default = 0)')
    parser.add_argument('--epochs', type=int, default=100, help='num epochs to train for')
    parser.add_argument('--lr', type=float, default=.001)
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    # 65 is all you need for GeoQuery
    parser.add_argument('--decoder_len_limit', type=int, default=65, help='output length limit of the decoder')
    parser.add_argument('--input_dim', type=int, default=100, help='input vector dimensionality')
    parser.add_argument('--output_dim', type=int, default=100, help='output vector dimensionality')
    parser.add_argument('--hidden_size', type=int, default=200, help='hidden state dimensionality')

    # Hyperparameters for the encoder -- feel free to play around with these!
    parser.add_argument('--no_bidirectional', dest='bidirectional', default=True, action='store_false', help='bidirectional LSTM')
    parser.add_argument('--no_reverse_input', dest='reverse_input', default=True, action='store_false', help='disable_input_reversal')
    parser.add_argument('--emb_dropout', type=float, default=0.2, help='input dropout rate')
    parser.add_argument('--rnn_dropout', type=float, default=0.2, help='dropout rate internal to encoder RNN')
    args = parser.parse_args()
    return args

def form_input(x):
    return torch.from_numpy(x).float()
# Semantic parser that uses Jaccard similarity to find the most similar input example to a particular question and
# returns the associated logical form.
class NearestNeighborSemanticParser(object):
    # Take any arguments necessary for parsing
    def __init__(self, training_data):
        self.training_data = training_data

    # decode should return a list of k-best lists of Derivations. A Derivation consists of the underlying Example,
    # a probability, and a tokenized output string. If you're just doing one-best decoding of example ex and you
    # produce output y_tok, you can just return the k-best list [Derivation(ex, 1.0, y_tok)]
    def decode(self, test_data):
        # Find the highest word overlap with the test data
        test_derivs = []
        for test_ex in test_data:
            test_words = test_ex.x_tok
            best_jaccard = -1
            best_train_ex = None
            for train_ex in self.training_data:
                # Compute word overlap
                train_words = train_ex.x_tok
                overlap = len(frozenset(train_words) & frozenset(test_words))
                jaccard = overlap/float(len(frozenset(train_words) | frozenset(test_words)))
                if jaccard > best_jaccard:
                    best_jaccard = jaccard
                    best_train_ex = train_ex
            # N.B. a list!
            test_derivs.append([Derivation(test_ex, 1.0, best_train_ex.y_tok)])
        return test_derivs


class Seq2SeqSemanticParser(object):
    def __init__(self,model_enc,model_dec,model_input_emb,model_output_emb,input_indexer, output_indexer,reverse_input ):
        self.model_enc = model_enc
        self.model_dec = model_dec
        self.model_input_emb = model_input_emb
        self.model_output_emb = model_output_emb
        self.input_indexer = input_indexer
        self.output_indexer = output_indexer
        self.reverse_input = reverse_input 
        #raise Exception("implement me!")
        # Add any args you need here

    def decode(self, test_data):
        self.model_enc.eval()
        self.model_dec.eval()
        self.model_input_emb.eval()
        self.model_output_emb.eval()
        input_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in test_data]))
        word_lens_tensor = torch.tensor(1).unsqueeze(0)
        all_test_input_data = make_padded_input_tensor(test_data, self.input_indexer, input_max_len, self.reverse_input)
        SOS_idx = torch.tensor(output_indexer.index_of(SOS_SYMBOL)).unsqueeze(0)
        SOS_embed = self.model_output_emb.forward(SOS_idx).unsqueeze(0)
        EOS_idx = torch.tensor(output_indexer.index_of(EOS_SYMBOL)).unsqueeze(0) 
        test_derivs = []
        
        for idx in range(len(test_data)):
            ex = test_data[idx]
            print(all_test_input_data[idx])
            x_tensor = torch.from_numpy(all_test_input_data[idx]).unsqueeze(0)
            inp_lens_tensor = torch.from_numpy(np.array(len(ex.x_tok))).unsqueeze(0)
            enc_word, b, context = encode_input_for_decoder(x_tensor, inp_lens_tensor, self.model_input_emb, self.model_enc)
            #word_idx_tensor = torch.tensor(self.output_indexer.index_of('<SOS>')).unsqueeze(0)
            hidden = context
            pred, hidden = self.model_dec.forward(SOS_embed, torch.tensor(1).unsqueeze(0), hidden, enc_word[:,:,00:])
            #pred, hidden = self.model_dec.forward(SOS_embed, torch.tensor(1).unsqueeze(0), hidden)
            hidden = (hidden[0].unsqueeze(0),hidden[1].unsqueeze(0))
            out_idx = torch.max(pred, dim = 1)[1]
            
            output=[]
            #first to last output words as input
            
            #for tgt_idx in ex.y_indexed:
            length = 0
            while out_idx != EOS_idx and length<200:
                output.append(self.output_indexer.get_object(out_idx.item()))
                word_emb = self.model_output_emb.forward(out_idx).unsqueeze(0)
                #print("shapes",word_emb.shape)
                #pred, hidden = self.model_dec.forward(word_emb, word_lens_tensor,hidden)
                pred, hidden = self.model_dec.forward(word_emb, word_lens_tensor,hidden, enc_word[:,:,00:])
                hidden = (hidden[0].unsqueeze(0), hidden[1].unsqueeze(0))
                out_idx = torch.max(pred, dim = 1)[1]
                print((out_idx))
                print(idx)
                #pred_idx_tensor = torch.tensor(pred_idx)
                #word_idx_tensor = pred_idx_tensor
                #print(pred_idx)
                length +=1
            test_derivs.append([Derivation(ex, 1.0, output)])
        return test_derivs


# Takes the given Examples and their input indexer and turns them into a numpy array by padding them out to max_len.
# Optionally reverses them.
def make_padded_input_tensor(exs, input_indexer, max_len, reverse_input):
    if reverse_input:
        return np.array(
            [[ex.x_indexed[len(ex.x_indexed) - 1 - i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
              for i in range(0, max_len)]
             for ex in exs])
    else:
        return np.array([[ex.x_indexed[i] if i < len(ex.x_indexed) else input_indexer.index_of(PAD_SYMBOL)
                          for i in range(0, max_len)]
                         for ex in exs])

# Analogous to make_padded_input_tensor, but without the option to reverse input
def make_padded_output_tensor(exs, output_indexer, max_len):
    return np.array([[ex.y_indexed[i] if i < len(ex.y_indexed) else output_indexer.index_of(PAD_SYMBOL) for i in range(0, max_len)] for ex in exs])

def len_input(exs):
    return np.array([[len(ex.x_indexed)] for ex in exs])
# Runs the encoder (input embedding layer and encoder as two separate modules) on a tensor of inputs x_tensor with
# inp_lens_tensor lengths.
# x_tensor: batch size x sent len tensor of input token indices
# inp_lens: batch size length vector containing the length of each sentence in the batch
# model_input_emb: EmbeddingLayer
# model_enc: RNNEncoder
# Returns the encoder outputs (per word), the encoder context mask (matrix of 1s and 0s reflecting

# E.g., calling this with x_tensor (0 is pad token):
# [[12, 25, 0, 0],
#  [1, 2, 3, 0],
#  [2, 0, 0, 0]]
# inp_lens = [2, 3, 1]
# will return outputs with the following shape:
# enc_output_each_word = 3 x 4 x dim, enc_context_mask = [[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]],
# enc_final_states = 3 x dim
def encode_input_for_decoder(x_tensor, inp_lens_tensor, model_input_emb, model_enc):
    input_emb = model_input_emb.forward(x_tensor)
    (enc_output_each_word, enc_context_mask, enc_final_states) = model_enc.forward(input_emb, inp_lens_tensor)
    enc_final_states_reshaped = (enc_final_states[0].unsqueeze(0), enc_final_states[1].unsqueeze(0))
    return (enc_output_each_word, enc_context_mask, enc_final_states_reshaped)


def train_model_encdec(train_data, test_data, input_indexer, output_indexer, args):
    # Sort in descending order by x_indexed, essential for pack_padded_sequence
    train_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)
    test_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)

    # Create indexed input
    input_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in train_data]))
    all_train_input_data = make_padded_input_tensor(train_data, input_indexer, input_max_len, args.reverse_input)
    all_test_input_data = make_padded_input_tensor(test_data, input_indexer, input_max_len, args.reverse_input)

    output_max_len = np.max(np.asarray([len(ex.y_indexed) for ex in train_data]))
    all_train_output_data = make_padded_output_tensor(train_data, output_indexer, output_max_len)
    all_test_output_data = make_padded_output_tensor(test_data, output_indexer, output_max_len)

    print("Train length: %i" % input_max_len)
    print("Train output length: %i" % np.max(np.asarray([len(ex.y_indexed) for ex in train_data])))
    print("Train matrix: %s; shape = %s" % (all_train_input_data, all_train_input_data.shape))
    print("x and y", train_data[0].x, train_data[0].x_indexed, train_data[0].y_tok, train_data[0].y_indexed)
    
    # Create model
    model_input_emb = EmbeddingLayer(args.input_dim, len(input_indexer), args.emb_dropout)
    model_enc = RNNEncoder(args.input_dim, args.hidden_size, args.rnn_dropout, args.bidirectional)
    
    model_output_emb = EmbeddingLayer(args.output_dim, len(output_indexer), args.emb_dropout)
    #model_dec = RNNDecoder(args.output_dim, args.hidden_size, len(output_indexer), dropout=0.15)
    model_dec = AttnRNNDecoder(args.output_dim, args.hidden_size, len(output_indexer),input_max_len, dropout=0.15)

    #model_dec = AttnDecoderRNN(args.output_dim, args.hidden_size, len(output_indexer), dropout_p=0.1, input_max_len)
    
    sos_ind = output_indexer.index_of(SOS_SYMBOL)
    eos_idx = output_indexer.index_of(EOS_SYMBOL)
    word_lens_tensor = torch.tensor(1).unsqueeze(0)
  
    #criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.NLLLoss()
    params = list(model_enc.parameters()) + list(model_dec.parameters()) + list(model_input_emb.parameters()) + list(model_output_emb.parameters())
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
            inp_lens_tensor = torch.from_numpy(np.array(len(train_data[idx].x_tok))).unsqueeze(0)
            enc_word, b, context = encode_input_for_decoder(x_tensor, inp_lens_tensor, model_input_emb, model_enc)
            first_idx_tensor = torch.tensor(sos_ind).unsqueeze(0)
            sos_embed = model_output_emb.forward(first_idx_tensor).unsqueeze(0)
            hidden = context
            
            #pred, hidden = model_dec.forward(sos_embed, word_lens_tensor,hidden)
            
            pred, hidden = model_dec.forward(sos_embed, word_lens_tensor,hidden,enc_word[:,:,00:])
            hidden = (hidden[0].unsqueeze(0), hidden[1].unsqueeze(0))
            #word_idx_tensor = first_idx_tensor 
            #print("input_size",len(context),inp_lens_tensor)
            #first to last output words as input
            
            for tgt_idx in train_data[idx].y_indexed:
                
                sent_loss += criterion(pred, torch.LongTensor([tgt_idx]))
                if tgt_idx == eos_idx:
                    break
                word_emb = model_output_emb.forward(torch.tensor(tgt_idx).unsqueeze(0)).unsqueeze(0)
                
                pred, hidden = model_dec.forward(word_emb, word_lens_tensor,hidden,enc_word[:,:,00:])
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



def train_model_encdec_beamsearch(train_data, test_data, input_indexer, output_indexer, args):
    # Sort in descending order by x_indexed, essential for pack_padded_sequence
    train_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)
    test_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)

    # Create indexed input
    input_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in train_data]))
    all_train_input_data = make_padded_input_tensor(train_data, input_indexer, input_max_len, args.reverse_input)
    all_test_input_data = make_padded_input_tensor(test_data, input_indexer, input_max_len, args.reverse_input)

    output_max_len = np.max(np.asarray([len(ex.y_indexed) for ex in train_data]))
    all_train_output_data = make_padded_output_tensor(train_data, output_indexer, output_max_len)
    all_test_output_data = make_padded_output_tensor(test_data, output_indexer, output_max_len)

    print("Train length: %i" % input_max_len)
    print("Train output length: %i" % np.max(np.asarray([len(ex.y_indexed) for ex in train_data])))
    print("Train matrix: %s; shape = %s" % (all_train_input_data, all_train_input_data.shape))
    print("x and y", train_data[0].x, train_data[0].x_indexed, train_data[0].y_tok, train_data[0].y_indexed)
    
    # Create model
    model_input_emb = EmbeddingLayer(args.input_dim, len(input_indexer), args.emb_dropout)
    model_enc = RNNEncoder(args.input_dim, args.hidden_size, args.rnn_dropout, args.bidirectional)
    
    model_output_emb = EmbeddingLayer(args.output_dim, len(output_indexer), args.emb_dropout)
    #model_dec = RNNDecoder(args.output_dim, args.hidden_size, len(output_indexer), dropout=0.15)
    model_dec = AttnRNNDecoder(args.output_dim, args.hidden_size, len(output_indexer),input_max_len, dropout=0.15)

    #model_dec = AttnDecoderRNN(args.output_dim, args.hidden_size, len(output_indexer), dropout_p=0.1, input_max_len)
    
    sos_ind = output_indexer.index_of(SOS_SYMBOL)
    eos_idx = output_indexer.index_of(EOS_SYMBOL)
    word_lens_tensor = torch.tensor(1).unsqueeze(0)
    logsoftmax = nn.LogSoftmax(dim=1)
    #criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.NLLLoss()
    params = list(model_enc.parameters()) + list(model_dec.parameters()) + list(model_input_emb.parameters()) + list(model_output_emb.parameters())
    optimizer = torch.optim.Adam(params, lr=0.0008)
    print("start training")
    for epoch in range(0,15):  #optimum 30
        model_enc.train()
        model_dec.train()
        model_input_emb.train()
        model_output_emb.train()
        
        perm0 =np.arange(len(train_data))
        #perm0 =np.arange(5)
        random.shuffle(perm0)
        for idx in perm0:
            
            model_enc.zero_grad()
            model_dec.zero_grad()
            model_input_emb.zero_grad()
            model_output_emb.zero_grad()
            sent_loss = torch.autograd.Variable(torch.FloatTensor([0]))
             
            #encoding input
            x_tensor = torch.from_numpy(all_train_input_data[idx]).unsqueeze(0)
            inp_lens_tensor = torch.from_numpy(np.array(len(train_data[idx].x_tok))).unsqueeze(0)
            enc_word, b, context = encode_input_for_decoder(x_tensor, inp_lens_tensor, model_input_emb, model_enc)
            first_idx_tensor = torch.tensor(sos_ind).unsqueeze(0)
            sos_embed = model_output_emb.forward(first_idx_tensor).unsqueeze(0)
            hidden = context
            #p#rint(hidden.shape)
            #print(enc_word.shape)
            pred, hidden = model_dec.forward(sos_embed, word_lens_tensor,hidden,enc_word[:,:,00:])
            hidden = (hidden[0].unsqueeze(0), hidden[1].unsqueeze(0))
            
            for tgt_idx in train_data[idx].y_indexed:
                
                sent_loss += criterion(pred, torch.LongTensor([tgt_idx]))
                if tgt_idx == eos_idx:
                    #print(torch.max(pred, dim = 1)[1],tgt_idx)
                    break
                word_emb = model_output_emb.forward(torch.tensor(tgt_idx).unsqueeze(0)).unsqueeze(0)       
                pred, hidden = model_dec.forward(word_emb, word_lens_tensor,hidden,enc_word[:,:,00:])
                hidden = (hidden[0].unsqueeze(0), hidden[1].unsqueeze(0))
            
            sent_loss.backward()
            #print(sent_loss, idx)
            #if sent_loss.item() > 10:
               #print(train_data[idx].y_tok)
            optimizer.step()
    
      # Create model
    model_input_emb = EmbeddingLayer(args.input_dim, len(input_indexer), args.emb_dropout)
    model_enc = RNNEncoder(args.input_dim, args.hidden_size, args.rnn_dropout, args.bidirectional)
    
    model_output_emb = EmbeddingLayer(args.output_dim, len(output_indexer), args.emb_dropout)
    #model_dec = RNNDecoder(args.output_dim, args.hidden_size, len(output_indexer), dropout=0.15)
    model_dec = AttnRNNDecoder(args.output_dim, args.hidden_size, len(output_indexer),input_max_len, dropout=0.15)

    print("finished training")
    for epoch in range(0,0):
        perm0 =np.arange(5)
        #perm0 =np.arange(len(train_data))
        random.shuffle(perm0)
        EOS_idx = torch.tensor(output_indexer.index_of(EOS_SYMBOL)).unsqueeze(0) 
        e = GeoqueryDomain()
        
        beam_size = 3
        beam = Beam(beam_size)
        model_enc.train()
        model_dec.train()
        model_input_emb.train()
        model_output_emb.train()
        loss = torch.autograd.Variable(torch.FloatTensor([0]))

        for idx in perm0:
            hyps = []
            ex = train_data[idx]
            pred_derivs =[]
            #encoding a sentence
            x_tensor = torch.from_numpy(all_train_input_data[idx]).unsqueeze(0)
            inp_lens_tensor = torch.from_numpy(np.array(len(ex.x_tok))).unsqueeze(0)
            enc_word, b, context = encode_input_for_decoder(x_tensor, inp_lens_tensor, model_input_emb, model_enc)
            hidden = context
            
            #decoding start sos
            sos_embed = model_output_emb.forward(first_idx_tensor).unsqueeze(0)
            #print("########################",sos_embed.shape)
            pred, hidden = model_dec.forward(sos_embed, torch.tensor(1).unsqueeze(0), hidden, enc_word[:,:,200:])
            hidden = (hidden[0].unsqueeze(0),hidden[1].unsqueeze(0))
            #log_prob = logsoftmax(pred)
            val,ind = torch.sort(pred[0],0)
            top_3 = ind[-3:]
            #store top 3 elements
            #print(val,ind)
            #print((log_prob[0][top_3[0]]))
            for k in range(0,3):
                #print(top_3[k])
                hyps.append(Hypothesis(tokens=[top_3[k]],
                                       log_probs=log_prob[0][top_3[k]],
                                       state=hidden,
                                       attn_dists=[],
                                       p_gens=[],
                                       coverage=[] # zero vector of length attention_length
                                       ) )
                #print(hyps[k].tokens,hyps[k].log_probs)
            results = []
            steps = 0
            #var = 3
            while steps < len(train_data[idx].y_indexed) and len(results) < beam_size:
                #latest_tokens = [h.latest_token for h in hyps] # latest token produced by each hypothesis
                #latest_tokens = [t if t in xrange(len(output_indexer)) else vocab.word2id(data.UNKNOWN_TOKEN) for t in latest_tokens] # change any in-article temporary OOV ids to [UNK] id, so that we can lookup word embeddings
                #states = [h.state for h in hyps] # list of current decoder states of the hypotheses
                var = 2*beam_size
                all_hyps = []
                for h in hyps:
                    latest_token = h.latest_token
                    hidden = h.state
                    #print ("************************",steps,h.tokens)
                    word_emb = model_output_emb.forward(torch.tensor(latest_token).unsqueeze(0)).unsqueeze(0)
                    #print(word_emb.shape)
                    pred, hidden = model_dec.forward(word_emb, word_lens_tensor,hidden, enc_word[:,:,200:])
                    hidden = (hidden[0].unsqueeze(0),hidden[1].unsqueeze(0))
                    log_prob = logsoftmax(pred)
                    val,ind = torch.sort(log_prob[0],0)
                    top_k = ind[-var:]
                    
                    for k in range(0,var):
                        #print(top_[k])
                        new_hyp = h.extend(token=[top_k[k]],
                                           log_prob=log_prob[0][top_k[k]],
                                           state=hidden,
                                           attn_dist=[],
                                           p_gen=[],
                                           coverage=[] # zero vector of length attention_length
                                           )
                        all_hyps.append(new_hyp)
                    
                hyps = []
                #print(all_hyps[0].avg_log_prob())
                #print("done")
                for h in sort_hyps(all_hyps): # in order of most likely h
                    if h.latest_token == eos_idx: # if stop token is reached...
                        # If this hypothesis is sufficiently long, put in results. Otherwise discard.
                        if steps >= 5:
                            results.append(h)
                    else: # hasn't reached stop token, so continue to extend this hypothesis
                        hyps.append(h)
                    if len(hyps) == beam_size or len(results) == beam_size:
                                # Once we've collected beam_size-many hypotheses for the next step, or beam_size-many complete hypotheses, stop.
                        break

                steps += 1
            if len(results)==0: # if we don't have any complete results, add all current hypotheses (incomplete summaries) to results
                results = hyps
            
            
            hyps_sorted = sort_hyps(results)
            pred_hypo = hyps_sorted[0]
            out_ind = pred_hypo.tokens
            output=[]
            for i in range(len(out_ind)):
                output.append(output_indexer.get_object(out_ind[i].item()))
                
            
            pred_derivs.append([Derivation(ex, 1.0, output)])
            extra, denotation_correct = e.compare_answers([train_data[idx].y], pred_derivs)
            #print("dddddddddddddd",train_data[idx].y, pred_derivs)
            reward = torch.tensor(0.0)
            if denotation_correct==True:
                reward = torch.tensor(1.0)
                print(reward)
            
            loss  += -1*reward*pred_hypo.log_probs
            #print("reqradddd",reward,pred_hypo.log_probs)
        loss.backward()
        optimizer.step()
        
        #print("finish", loss.item())  
        #print(denotation_correct)
    #break
    return Seq2SeqSemanticParser(model_enc,model_dec,model_input_emb,model_output_emb,input_indexer,output_indexer,args.reverse_input )
def sort_hyps(hyps):
    """#Return a list of Hypothesis objects, sorted by descending average log probability"""
    return sorted(hyps, key=lambda h: h.avg_log_prob(), reverse=True)
# Evaluates decoder against the data in test_data (could be dev data or test data). Prints some output
# every example_freq examples. Writes predictions to outfile if defined. Evaluation requires
# executing the model's predictions against the knowledge base. We pick the highest-scoring derivation for each
# example with a valid denotation (if you've provided more than one).
def evaluate(test_data, decoder, example_freq=50, print_output=True, outfile=None):
    e = GeoqueryDomain()
    pred_derivations = decoder.decode(test_data)
    
    selected_derivs, denotation_correct = e.compare_answers([ex.y for ex in test_data], pred_derivations)
    num_exact_match = 0
    num_tokens_correct = 0
    num_denotation_match = 0
    total_tokens = 0
    for i, ex in enumerate(test_data):
        if i % example_freq == 0:
            print('Example %d' % i)
            print('  x      = "%s"' % ex.x)
            print('  y_tok  = "%s"' % ex.y_tok)
            print('  y_pred = "%s"' % selected_derivs[i].y_toks)
        # Compute accuracy metrics
        y_pred = ' '.join(selected_derivs[i].y_toks)
        # Check exact match
        if y_pred == ' '.join(ex.y_tok):
            num_exact_match += 1
        # Check position-by-position token correctness
        num_tokens_correct += sum(a == b for a, b in zip(selected_derivs[i].y_toks, ex.y_tok))
        total_tokens += len(ex.y_tok)
        # Check correctness of the denotation
        if denotation_correct[i]:
            num_denotation_match += 1
    if print_output:
        print("Exact logical form matches: %s" % (render_ratio(num_exact_match, len(test_data))))
        print("Token-level accuracy: %s" % (render_ratio(num_tokens_correct, total_tokens)))
        print("Denotation matches: %s" % (render_ratio(num_denotation_match, len(test_data))))
    # Writes to the output file if needed
    if outfile is not None:
        with open(outfile, "w") as out:
            for i, ex in enumerate(test_data):
                out.write(ex.x + "\t" + " ".join(selected_derivs[i].y_toks) + "\n")
        out.close()
    print("########################",pred_derivations)

def render_ratio(numer, denom):
    return "%i / %i = %.3f" % (numer, denom, float(numer)/denom)


if __name__ == '__main__':
    args = _parse_args()
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # Load the training and test data

    train, dev, test = load_datasets(args.train_path, args.dev_path, args.test_path, domain=args.domain)
    train_data_indexed, dev_data_indexed, test_data_indexed, input_indexer, output_indexer = index_datasets(train, dev, test, args.decoder_len_limit)
    print("%i train exs, %i dev exs, %i input types, %i output types" % (len(train_data_indexed), len(dev_data_indexed), len(input_indexer), len(output_indexer)))
    print("Input indexer: %s" % input_indexer)
    print("Output indexer: %s" % output_indexer)
    print("Here are some examples post tokenization and indexing:")
    for i in range(0, min(len(train_data_indexed), 10)):
        print(train_data_indexed[i])
    if args.do_nearest_neighbor:
        decoder = NearestNeighborSemanticParser(train_data_indexed)
        evaluate(dev_data_indexed, decoder)
    else:
        #decoder = train_model_encdec(train_data_indexed, dev_data_indexed, input_indexer, output_indexer, args)
        decoder = train_model_encdec_beamsearch(train_data_indexed, dev_data_indexed, input_indexer, output_indexer, args)

        evaluate(dev_data_indexed, decoder)
    print("=======FINAL EVALUATION ON BLIND TEST=======")
    #evaluate(test_data_indexed, decoder, print_output=False, outfile="geo_test_output.tsv")


