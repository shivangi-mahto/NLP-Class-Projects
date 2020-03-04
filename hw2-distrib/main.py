import argparse
import random
import numpy as np
import time
import torch
from torch import optim
from lf_evaluator import *
from models import *
from data import *
from utils import *
import math
from sentiment_data import *


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
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    # 65 is all you need for GeoQuery
    parser.add_argument('--decoder_len_limit', type=int, default=65, help='output length limit of the decoder')
    parser.add_argument('--input_dim', type=int, default=100, help='input vector dimensionality')
    parser.add_argument('--output_dim', type=int, default=100, help='output vector dimensionality')
    parser.add_argument('--hidden_size', type=int, default=200, help='hidden state dimensionality')
    parser.add_argument('--word_vecs_path', type=str, default='data/glove.6B.300d-relativized.txt', help='path to word vectors file')

    # Hyperparameters for the encoder -- feel free to play around with these!
    parser.add_argument('--no_bidirectional', dest='bidirectional', default=True, action='store_false', help='bidirectional LSTM')
    parser.add_argument('--no_reverse_input', dest='reverse_input', default=True, action='store_false', help='disable_input_reversal')
    parser.add_argument('--emb_dropout', type=float, default=0.2, help='input dropout rate')
    parser.add_argument('--rnn_dropout', type=float, default=0.2, help='dropout rate internal to encoder RNN')
    args = parser.parse_args()
    return args

#Seq2SeqSemanticParser has both decoders 
#
#for attention based training with scheduled sampling go to line 284
#or 30 epochs in Attention model without scheduled sampling and greedy decoder
#exact logical form matches: 70 / 120 = 0.583
#Token-level accuracy: 3194 / 3908 = 0.817
#Denotation matches: 76 / 120 = 0.633

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
    
    def __init__(self, input_indexer, output_indexer, model_enc,model_dec,model_input_emb,model_output_emb,reverse_input,output_max_len):
        self.input_indexer = input_indexer
        self.output_indexer = output_indexer
        self.model_enc = model_enc
        self.model_dec = model_dec
        self.model_input_emb = model_input_emb
        self.model_output_emb = model_output_emb
        self.reverse_input = reverse_input
        self.output_max_len = output_max_len

        #raise Exception("implement me!")
        # Add any args you need here

    def decode(self, test_data):
        self.model_enc.eval()
        self.model_dec.eval()
        self.model_input_emb.eval()
        self.model_output_emb.eval()
        
        sos_idx = self.output_indexer.index_of(SOS_SYMBOL)
        eos_idx = self.output_indexer.index_of(EOS_SYMBOL)
        test_derivs =[]
        input_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in test_data]))
        all_test_input_data = make_padded_input_tensor(test_data, self.input_indexer, input_max_len, self.reverse_input)

        for idx in range(len(test_data)):
            ex = test_data[idx]
            #print("ccc",ex.x_tok)
            #print("rrrr",ex.y_tok)
            input_sent = torch.from_numpy(all_test_input_data[idx]).unsqueeze(0)
            inp_lens_tensor = torch.from_numpy(np.array(len(test_data[idx].x_indexed))).unsqueeze(0)
            enc_word, b, context = encode_input_for_decoder(input_sent, inp_lens_tensor, self.model_input_emb, self.model_enc)

            sos_embed = self.model_output_emb.forward(torch.tensor(sos_idx).unsqueeze(0))
            pred, hidden = self.model_dec.forward(sos_embed.unsqueeze(0), context, enc_word)
            #print("predshape",pred.squeeze(0).shape)
            out_idx = torch.max(pred.squeeze(0), dim = 1)[1]
            #print(out_idx)
            length = 0
            output_seq = []
            while out_idx!=eos_idx and length < 3*self.output_max_len:
                #print(out_idx.item())
                output_seq.append(self.output_indexer.get_object(out_idx.item()))
                word_embed = self.model_output_emb.forward(torch.tensor(out_idx.item()).unsqueeze(0))
                pred, hidden = self.model_dec.forward(word_embed.unsqueeze(0), hidden, enc_word)
                out_idx = torch.max(pred.squeeze(0), dim = 1)[1]
                length += 1
            test_derivs.append([Derivation(ex, 1.0, output_seq)])
        return test_derivs
    
    def beamsearch_decode(self, test_data):
        self.model_enc.eval()
        self.model_dec.eval()
        self.model_input_emb.eval()
        self.model_output_emb.eval()
        beam_size = 9
        sos_idx = self.output_indexer.index_of(SOS_SYMBOL)
        eos_idx = self.output_indexer.index_of(EOS_SYMBOL)
        test_derivs =[]
        
        input_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in test_data]))
        all_test_input_data = make_padded_input_tensor(test_data, self.input_indexer, input_max_len, self.reverse_input)

        for idx in range(len(test_data)):#
            ex = test_data[idx]
            hyps=[]
            input_sent = torch.from_numpy(all_test_input_data[idx]).unsqueeze(0)
            inp_lens_tensor = torch.from_numpy(np.array(len(test_data[idx].x_indexed))).unsqueeze(0)
            enc_word, b, context = encode_input_for_decoder(input_sent, inp_lens_tensor, self.model_input_emb, self.model_enc)

            sos_embed = self.model_output_emb.forward(torch.tensor(sos_idx).unsqueeze(0))
            pred, hidden = self.model_dec.forward(sos_embed.unsqueeze(0), context, enc_word)
            val,ind = torch.sort(pred.squeeze(0)[0],0)
            #print(val,ind)
            top_3 = ind[-beam_size:] # top3 prediction from beamsearch
            for k in range(0,beam_size): #store 3 index hyptothesis
                hyps.append(Hypothesis(tokens=[top_3[k]], log_probs=pred.squeeze(0)[0][top_3[k]], state=hidden, attn_dists=[],p_gens=[], coverage=[] ) )
            
            results = []
            length = 0
            while length < 3*self.output_max_len and len(results) < beam_size:
                var = 2*beam_size
                all_hyps = []
                for h in hyps:
                    latest_token = h.latest_token
                    hidden = h.state
                    word_emb = self.model_output_emb.forward(torch.tensor(latest_token).unsqueeze(0))
                    pred, hidden = self.model_dec.forward(word_emb.unsqueeze(0), hidden, enc_word)
                    val,ind = torch.sort(pred.squeeze(0)[0],0)
                    top_k = ind[-var:]
                 
                    for k in range(0,var):
                        new_hyp = h.extend(token=[top_k[k]],log_prob=pred.squeeze(0)[0][top_k[k]],state=hidden,attn_dist=[],p_gen=[],coverage=[])
                        all_hyps.append(new_hyp)
                hyps = []
                for h in sort_hyps(all_hyps): # in order of most likely h
                    if h.latest_token == eos_idx: # if stop token is reached...
                        if length >= 7: #7 is best
                            h.tokens = h.tokens[:-1]
                            results.append(h)
                    else: 
                        hyps.append(h)
                    if len(hyps) == beam_size or len(results) == beam_size:
                        break

                length += 1
            if len(results)==0: # if we don't have any complete results, add all current hypotheses (incomplete summaries) to results
                results = hyps
            
            hyps_sorted = sort_hyps(results)
            out_ind = hyps_sorted[0].tokens
            output_seq=[]
            for i in range(len(out_ind)):
                output_seq.append(self.output_indexer.get_object(out_ind[i].item()))
                
            test_derivs.append([Derivation(ex, 1.0, output_seq)])
        return test_derivs

def sort_hyps(hyps):
    """#Return a list of Hypothesis objects, sorted by descending average log probability"""
    return sorted(hyps, key=lambda h: h.avg_log_prob(), reverse=True)

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

    # Create model
    model_input_emb = EmbeddingLayer(args.input_dim, len(input_indexer), args.emb_dropout)
    model_enc = RNNEncoder(args.input_dim, args.hidden_size, args.rnn_dropout, args.bidirectional)
    model_output_emb = EmbeddingLayer(args.output_dim, len(output_indexer), args.emb_dropout)
    model_dec = LSTMAttnClassifier(args.output_dim, args.hidden_size, len(output_indexer))
    
    sos_idx = output_indexer.index_of(SOS_SYMBOL)
    eos_idx = output_indexer.index_of(EOS_SYMBOL)
    criterion = torch.nn.NLLLoss()
    params = list(model_enc.parameters()) + list(model_dec.parameters()) + list(model_input_emb.parameters()) + list(model_output_emb.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001)
    # Loop over epochs, loop over examples, given some indexed words, call encode_input_for_decoder, then call your
    # decoder, accumulate losses, update parameters
    model_enc.train()
    model_dec.train()
    model_input_emb.train()
    model_output_emb.train()
    count = 0 
    for epoch in range(0,1):#20
        perm0 =np.arange(len(train_data))
        
        random.shuffle(perm0)
        for idx in perm0:
            
            truth_label = []
            pred_label = []
            
            teacher_forcing_ratio = (0.999)**(count)
            optimizer.zero_grad()
            
            sent_loss = 0
           
            input_sent = torch.from_numpy(all_train_input_data[idx]).unsqueeze(0)
            inp_lens_tensor = torch.from_numpy(np.array(len(train_data[idx].x_indexed))).unsqueeze(0)
            enc_word, b, context = encode_input_for_decoder(input_sent, inp_lens_tensor, model_input_emb, model_enc)
           
            inp_idx = sos_idx
            hidden = context
            #"""This is simple training, For training scheduled sampling - make this False n below True
            if True:
                for tgt_idx in train_data[idx].y_indexed:
                    word_embed = model_output_emb.forward(torch.tensor(inp_idx).unsqueeze(0))
                    pred, hidden = model_dec.forward(word_embed.unsqueeze(0), hidden, enc_word)
                    sent_loss +=  criterion(pred.squeeze(0), torch.LongTensor([tgt_idx]))
                    truth_label.append(tgt_idx)
                    pred_label.append(torch.max(pred.squeeze(0), dim = 1)[1].item())
                    inp_idx = tgt_idx
                    if tgt_idx == eos_idx:
                        break
                
            if False: # Make true for scheduled sampling 
                use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    
                for tgt_idx in train_data[idx].y_indexed:
                    
                    word_embed = model_output_emb.forward(torch.tensor(inp_idx).unsqueeze(0))
                    pred, hidden = model_dec.forward(word_embed.unsqueeze(0), hidden, enc_word)
                    sent_loss +=  criterion(pred.squeeze(0), torch.LongTensor([tgt_idx]))
                    if use_teacher_forcing:
                        inp_idx = tgt_idx
                    else:
                        inp_idx = torch.max(pred.squeeze(0), dim = 1)[1].item()
                        
                    if inp_idx == eos_idx:
                        break
            count = count +1            
            sent_loss.backward()
            optimizer.step()
    print ("finished training")  
    path_to_input_emb = "./input_emb.pth" 
    path_to_output_emb = "./output_emb.pth"
    path_to_enc = "./enc.pth"
    path_to_dec = "./dec.pth"
    torch.save(model_input_emb.state_dict(), path_to_input_emb)
    torch.save(model_output_emb.state_dict(), path_to_output_emb)
    torch.save(model_enc.state_dict(), path_to_enc)
    torch.save(model_dec.state_dict(), path_to_dec)
    
    return Seq2SeqSemanticParser(input_indexer, output_indexer, model_enc,model_dec,model_input_emb,model_output_emb,args.reverse_input,output_max_len)
    #return Seq2SeqSemanticParser(model_enc,model_dec,model_input_emb,model_output_emb,input_indexer,output_indexer,args.reverse_input)


def train_model_encdec_self(train_data, test_data, input_indexer, output_indexer, args):
    # Sort in descending order by x_indexed, essential for pack_padded_sequence
    train_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)
    test_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)
   
    # Create indexed input
    input_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in train_data]))
    all_train_input_data = make_padded_input_tensor(train_data, input_indexer, input_max_len, args.reverse_input)
    all_test_input_data = make_padded_input_tensor(test_data, input_indexer, input_max_len, args.reverse_input)

    #output_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in train_data]))
    #all_train_output_data = make_padded_output_tensor(train_data, output_indexer, output_max_len)
    #all_test_output_data = make_padded_output_tensor(test_data, output_indexer, output_max_len)

    print("Train length: %i" % input_max_len)
    print("Train output length: %i" % np.max(np.asarray([len(ex.y_indexed) for ex in train_data])))
    print("Train matrix: %s; shape = %s" % (all_train_input_data, all_train_input_data.shape))

    # Create model
    model_input_emb = EmbeddingLayer(args.input_dim, len(input_indexer), args.emb_dropout)
    model_enc = RNNEncoder(args.input_dim, args.hidden_size, args.rnn_dropout, args.bidirectional)
    model_output_emb = EmbeddingLayer(args.output_dim, len(output_indexer), args.emb_dropout)
    model_dec = LSTMAttnClassifier(args.output_dim, args.hidden_size, len(output_indexer))
    
    sos_idx = output_indexer.index_of(SOS_SYMBOL)
    eos_idx = output_indexer.index_of(EOS_SYMBOL)
    criterion = torch.nn.NLLLoss()
    params = list(model_enc.parameters()) + list(model_dec.parameters()) + list(model_input_emb.parameters()) + list(model_output_emb.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001)
    # Loop over epochs, loop over examples, given some indexed words, call encode_input_for_decoder, then call your
    # decoder, accumulate losses, update parameters
    model_enc.train()
    model_dec.train()
    model_input_emb.train()
    model_output_emb.train()
    count = 0 
    for epoch in range(0,1):#20
        perm0 =np.arange(len(train_data))
        
        random.shuffle(perm0)
        for idx in perm0:
            
            truth_label = []
            pred_label = []
            
            teacher_forcing_ratio = (0.999)**(count)
            optimizer.zero_grad()
            
            sent_loss = 0
           
            input_sent = torch.from_numpy(all_train_input_data[idx]).unsqueeze(0)
            inp_lens_tensor = torch.from_numpy(np.array(len(train_data[idx].x_indexed))).unsqueeze(0)
            enc_word, b, context = encode_input_for_decoder(input_sent, inp_lens_tensor, model_input_emb, model_enc)
           
            inp_idx = sos_idx
            hidden = context
            #"""This is simple training, For training scheduled sampling - make this False n below True
            if True:
                for tgt_idx in train_data[idx].x_indexed:
                    word_embed = model_output_emb.forward(torch.tensor(inp_idx).unsqueeze(0))
                    pred, hidden = model_dec.forward(word_embed.unsqueeze(0), hidden, enc_word)
                    sent_loss +=  criterion(pred.squeeze(0), torch.LongTensor([tgt_idx]))
                    truth_label.append(tgt_idx)
                    pred_label.append(torch.max(pred.squeeze(0), dim = 1)[1].item())
                    inp_idx = tgt_idx
                    if tgt_idx == eos_idx:
                        break
            
            count = count +1            
            sent_loss.backward()
            optimizer.step()
    print ("finished training")  
    path_to_input_emb = "./input_emb.pth" 
    path_to_output_emb = "./output_emb.pth"
    path_to_enc = "./enc.pth"
    path_to_dec = "./dec.pth"
    torch.save(model_input_emb.state_dict(), path_to_input_emb)
    torch.save(model_output_emb.state_dict(), path_to_output_emb)
    torch.save(model_enc.state_dict(), path_to_enc)
    torch.save(model_dec.state_dict(), path_to_dec)
    
    return Seq2SeqSemanticParser(input_indexer, output_indexer, model_enc,model_dec,model_input_emb,model_output_emb,args.reverse_input,output_max_len)
    #return Seq2SeqSemanticParser(model_enc,model_dec,model_input_emb,model_output_emb,input_indexer,output_indexer,args.reverse_input)


# if you want to fine tune models with denotation prediction training as GUU2017 did! we dont have results for this!
def train_model_beamsearch_encdec(train_data, test_data, input_indexer, output_indexer, args):
    # Sort in descending order by x_indexed, essential for pack_padded_sequence
    train_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)
    test_data.sort(key=lambda ex: len(ex.x_indexed), reverse=True)
    Vector_dict = word_vectors.vectors
    e = GeoqueryDomain()
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
    
    sos_idx = output_indexer.index_of(SOS_SYMBOL)
    eos_idx = output_indexer.index_of(EOS_SYMBOL)
    criterion = torch.nn.NLLLoss() 
    
    model_input_emb_imp = EmbeddingLayer(args.input_dim, len(input_indexer), args.emb_dropout)
    model_enc_imp = RNNEncoder(args.input_dim, args.hidden_size, args.rnn_dropout, args.bidirectional)
    model_output_emb_imp = EmbeddingLayer(args.output_dim, len(output_indexer), args.emb_dropout)
    model_dec_imp = LSTMAttnClassifier(args.output_dim, args.hidden_size, len(output_indexer))

    path_to_input_emb = "./input_emb_65_den_acc.pth" 
    path_to_output_emb = "./output_emb_65_den_acc.pth"
    path_to_enc = "./enc_65_den_acc.pth"
    path_to_dec = "./dec_65_den_acc.pth"
    
    model_input_emb_imp.load_state_dict(torch.load(path_to_input_emb))
    model_output_emb_imp.load_state_dict(torch.load(path_to_output_emb))
    model_enc_imp.load_state_dict(torch.load(path_to_enc))
    model_dec_imp.load_state_dict(torch.load(path_to_dec))
    
    params = list(model_enc_imp.parameters()) + list(model_dec_imp.parameters()) + list(model_input_emb_imp.parameters()) + list(model_output_emb_imp.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001)
    # Loop over epochs, loop over examples, given some indexed words, call encode_input_for_decoder, then call your
    # decoder, accumulate losses, update parameters
    model_enc_imp.train()
    model_dec_imp.train()
    model_input_emb_imp.train()
    model_output_emb_imp.train()
    
    for epoch in range(0,1):
        optimizer.zero_grad()
        beam_size = 3
        
        count = 0
        print("count",0)
        input_max_len = np.max(np.asarray([len(ex.x_indexed) for ex in train_data]))
        all_train_input_data = make_padded_input_tensor(train_data, input_indexer, input_max_len, args.reverse_input)

        loss = torch.autograd.Variable(torch.FloatTensor([0]))
        perm0 =np.arange(len(train_data))
        #perm0 =np.arange(5)
        random.shuffle(perm0)
        #for idx in perm0:
        for idx in range(4):#
            
            ex = train_data[idx]
            hyps=[]
            input_sent = torch.from_numpy(all_train_input_data[idx]).unsqueeze(0)
            inp_lens_tensor = torch.from_numpy(np.array(len(train_data[idx].x_indexed))).unsqueeze(0)
            enc_word, b, context = encode_input_for_decoder(input_sent, inp_lens_tensor, model_input_emb_imp, model_enc_imp)

            sos_embed = model_output_emb_imp.forward(torch.tensor(sos_idx).unsqueeze(0))
            pred, hidden = model_dec_imp.forward(sos_embed.unsqueeze(0), context, enc_word)
            val,ind = torch.sort(pred.squeeze(0)[0],0)
            #print(val,ind)
            top_3 = ind[-beam_size:] # top3 prediction from beamsearch
            #print(top_3[2],pred.squeeze(0)[0][top_3[2]])
            for k in range(0,beam_size): #store 3 index hyptothesis
                hyps.append(Hypothesis(tokens=[top_3[k]], log_probs=pred.squeeze(0)[0][top_3[k]], state=hidden, attn_dists=[],p_gens=[], coverage=[] ) )
            #print(hyps[0].tokens,hyps[0].log_probs )
            results = []
            length = 0
            while length < 3*output_max_len and len(results) < beam_size:
                var = 2*beam_size
                all_hyps = []
                for h in hyps:
                    latest_token = h.latest_token
                    hidden = h.state
                    word_emb = model_output_emb_imp.forward(torch.tensor(latest_token).unsqueeze(0))
                    pred, hidden = model_dec_imp.forward(word_emb.unsqueeze(0), hidden, enc_word)
                    val,ind = torch.sort(pred.squeeze(0)[0],0)
                    top_k = ind[-var:]
                    
                    for k in range(0,var):
                        new_hyp = h.extend(token=[top_k[k]],log_prob=pred.squeeze(0)[0][top_k[k]],state=hidden,attn_dist=[],p_gen=[],coverage=[])
                        all_hyps.append(new_hyp)
                    
                hyps = []
                for h in sort_hyps(all_hyps): # in order of most likely h
                    #print("h token and eos ",h.latest_token, eos_idx)
                    if h.latest_token == eos_idx: # if stop token is reached...
                        if length >= 7:
                        #print("reached eos_idx")
                            h.tokens = h.tokens[:-1]
                            results.append(h)
                        #print(h.tokens,h.log_probs )
                    else: 
                        hyps.append(h)
                        #print(hyps[0].tokens,hyps[0].log_probs )
                    if len(hyps) == beam_size or len(results) == beam_size:
                        break

                length += 1
                
            if len(results)==0: # if we don't have any complete results, add all current hypotheses (incomplete summaries) to results
                results = hyps
            
            hyps_sorted = sort_hyps(results)
            for j in range(0,1):
                test_derivs =[]
                out_ind = hyps_sorted[j].tokens
                output_seq=[]
                for i in range(len(out_ind)):
                    output_seq.append(output_indexer.get_object(out_ind[i].item()))
                
                test_derivs.append([Derivation(ex, 1.0, output_seq)])
                print("testderivs",test_derivs)
                print("traidata",[train_data[idx].y_tok])
                extra, denotation_correct = e.compare_answers(train_data[idx].y_tok, test_derivs)
                count=count+1
                print(count)
                if denotation_correct == True:
                    loss +=  -1*hyps_sorted[j].log_probs
                #else: 
                    #loss +=  1*hyps_sorted[j].log_probs
        loss.backward()
        optimizer.step()
        
    path_to_input_emb = "./input_emb_imp.pth" 
    path_to_output_emb = "./output_emb_imp.pth"
    path_to_enc = "./enc_imp.pth"
    path_to_dec = "./dec_imp.pth"
    #raise Exception("Implement the rest of me to train your parser")
    torch.save(model_input_emb_imp.state_dict(), path_to_input_emb)
    torch.save(model_output_emb_imp.state_dict(), path_to_output_emb)
    torch.save(model_enc_imp.state_dict(), path_to_enc)
    torch.save(model_dec_imp.state_dict(), path_to_dec)
    #raise Exception("Implement the rest of me to train your parser")
    return Seq2SeqSemanticParser(input_indexer, output_indexer, model_enc_imp,model_dec_imp,model_input_emb_imp,model_output_emb_imp,args.reverse_input,output_max_len)
    #return Seq2SeqSemanticParser(model_enc,model_dec,model_input_emb,model_output_emb,input_indexer,output_indexer,args.reverse_input)


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

def evaluate_bs(test_data, decoder, example_freq=50, print_output=True, outfile=None):
    e = GeoqueryDomain()
    pred_derivations = decoder.beamsearch_decode(test_data) 
    selected_derivs, denotation_correct = e.compare_answers([ex.y for ex in test_data], pred_derivations)
    num_exact_match = 0
    num_tokens_correct = 0
    num_denotation_match = 0
    total_tokens = 0
    for i, ex in enumerate(test_data):
        if i % example_freq == 0:
            print('Example %d' % i)
            #print('  x      = "%s"' % ex.x)
            #print('  y_tok  = "%s"' % ex.y_tok)
            #print('  y_pred = "%s"' % selected_derivs[i].y_toks)
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
    #print("%i train exs, %i dev exs, %i input types, %i output types" % (len(train_data_indexed), len(dev_data_indexed), len(input_indexer), len(output_indexer)))
    #print("Input indexer: %s" % input_indexer)
    #print("Output indexer: %s" % output_indexer)
    #print("Here are some examples post tokenization and indexing:")
    for i in range(0, min(len(train_data_indexed), 10)):
        print(train_data_indexed[i])
    if args.do_nearest_neighbor:
        decoder = NearestNeighborSemanticParser(train_data_indexed)
        evaluate(dev_data_indexed, decoder)
    else:#decoder with attention and schedule sampling
        #decoder =  train_model_encdec_self(train_data_indexed, dev_data_indexed, input_indexer, output_indexer, args)
        
        decoder =  train_model_encdec_self(train_data_indexed, dev_data_indexed, input_indexer, input_indexer, args)

        #to run implementation of finetuning 
        #decoder =  train_model_beamsearch_encdec(train_data_indexed, dev_data_indexed, input_indexer, output_indexer, args) 
        #to evaluate using beamsearch decoder
        #evaluate_bs(dev_data_indexed, decoder)
        #to evaluate using greedy decoder
        evaluate(dev_data_indexed, decoder) #to evalaute beam search use this evaluate_bs(dev_data_indexed, decoder)
    print("=======FINAL EVALUATION ON BLIND TEST=======")
    evaluate(test_data_indexed, decoder, print_output=False, outfile="geo_test_output.tsv")
    

