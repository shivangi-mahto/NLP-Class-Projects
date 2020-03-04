import argparse
import random
import numpy as np
import time
import torch
from torch import optim
from lf_evaluator import *
from model_second import *
from data import *
from utils import *
import math
from torchvision import models
#from torchsummary import summary
#from pyrouge import Rouge155
#from sentiment_data import *

PAD_SYMBOL = "<PAD>"
SOS_SYMBOL = "<SOS>"
EOS_SYMBOL = "<EOS>"
UNK_SYMBOL = "<UNK>"

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
    parser.add_argument('--input_dim', type=int, default=300, help='input vector dimensionality')
    parser.add_argument('--output_dim', type=int, default=300, help='output vector dimensionality')
    parser.add_argument('--hidden_size', type=int, default=200, help='hidden state dimensionality')
    #parser.add_argument('--word_vecs_path', type=str, default='data/glove.6B.300d-relativized.txt', help='path to word vectors file')
    parser.add_argument('--word_vecs_path', type=str, default='glove.6B.300d-relativized.txt', help='path to word vectors file')
    parser.add_argument('--para_ques_url', type=str, default='https://trec.nist.gov/data/qa/2017_LiveQA/questionParaphrases.xml')
    parser.add_argument('--answer_file', type=str, default='./Answer_file.txt')

    # Hyperparameters for the encoder -- feel free to play around with these!
    parser.add_argument('--no_bidirectional', dest='bidirectional', default=False, action='store_false', help='bidirectional LSTM')
    parser.add_argument('--no_reverse_input', dest='reverse_input', default=False, action='store_false', help='disable_input_reversal')
    parser.add_argument('--emb_dropout', type=float, default=0.2, help='input dropout rate')
    parser.add_argument('--rnn_dropout', type=float, default=0.2, help='dropout rate internal to encoder RNN')
    args = parser.parse_args()
    return args

#Seq2SeqSemanticParser has both decoders 


class Summarizer(object):
    
    def __init__(self, input_indexer, output_indexer, model_enc,model_dec,model_input_emb,model_output_emb,reverse_input,output_max_len, device):
        self.input_indexer = input_indexer
        self.output_indexer = output_indexer
        self.model_enc = model_enc
        self.model_dec = model_dec
        self.model_input_emb = model_input_emb
        self.model_output_emb = model_output_emb
        self.reverse_input = reverse_input
        self.output_max_len = output_max_len
        self.device = device
        #raise Exception("implement me!")
        # Add any args you need here

    def decode(self, test_data):
        self.model_enc.eval()
        self.model_dec.eval()
        self.model_input_emb.eval()
        self.model_output_emb.eval()
        f= open("answers_simple_decode.txt","w")
        g= open("answers_original_decode.txt","w")
        sos_idx = self.output_indexer.index_of(SOS_SYMBOL)
        eos_idx = self.output_indexer.index_of(EOS_SYMBOL)
        test_derivs =[]
        input_max_len = np.max(np.asarray([len(ex.doc) for ex in test_data]))
        all_test_input_data = make_padded_input_tensor(test_data, self.input_indexer, input_max_len, self.reverse_input)

        for idx in range(len(test_data)):
            ex = test_data[idx]
            
            input_sent = torch.from_numpy(all_test_input_data[idx]).unsqueeze(0).to(self.device)
            inp_lens_tensor = torch.from_numpy(np.array(len(test_data[idx].doc))).unsqueeze(0).to(self.device)
            enc_word, b, context = encode_input_for_decoder(input_sent, inp_lens_tensor, self.model_input_emb, self.model_enc, self.device)

            sos_embed = self.model_output_emb.forward(torch.tensor(sos_idx).unsqueeze(0).to(self.device))
            pred, hidden = self.model_dec.forward(sos_embed.unsqueeze(0), context, enc_word)
            
            out_idx = torch.max(pred.squeeze(0), dim = 1)[1]
            
            length = 0
            original_ans = []
            output_seq = []
            while out_idx!=eos_idx and length < self.output_max_len:
                
                output_seq.append(self.output_indexer.get_object(out_idx.item()))
                if(length<(len(ex.best_x))):
                    original_ans.append(self.output_indexer.get_object(ex.best_x[length]))
                word_embed = self.model_output_emb.forward(torch.tensor(out_idx.item()).unsqueeze(0).to(self.device))
                pred, hidden = self.model_dec.forward(word_embed.unsqueeze(0), hidden, enc_word)
                out_idx = torch.max(pred.squeeze(0), dim = 1)[1]
                length += 1
            print(output_seq)
            joint_ans = " ".join(output_seq)
            
            joint_orig = " ".join(original_ans)
            print("joint answers are", joint_ans)
            print("joint original are", joint_orig)
            print(type(joint_ans))
            
            f.write(joint_ans+"\n")
            g.write(joint_orig+"\n")
            #test_derivs.append([Derivation(ex, 1.0, output_seq)])
        f.close()
        g.close()
        return test_derivs
    
    def beamsearch_decode(self, test_data):
        self.model_enc.eval()
        self.model_dec.eval()
        self.model_input_emb.eval()
        self.model_output_emb.eval()
        f= open("answers_beamsearch.txt","w")
        g= open("answers_original_decode.txt","w")
        beam_size = 9
        sos_idx = self.output_indexer.index_of(SOS_SYMBOL)
        eos_idx = self.output_indexer.index_of(EOS_SYMBOL)
        test_derivs =[]
        print("running beam search decode")   
        input_max_len = np.max(np.asarray([len(ex.doc) for ex in test_data]))
        all_test_input_data = make_padded_input_tensor(test_data, self.input_indexer, input_max_len, self.reverse_input)

        for idx in range(len(test_data)):#
            ex = test_data[idx]
            hyps=[]
            input_sent = torch.from_numpy(all_test_input_data[idx]).unsqueeze(0).to(self.device)
            inp_lens_tensor = torch.from_numpy(np.array(len(test_data[idx].doc))).unsqueeze(0).to(self.device)
            enc_word, b, context = encode_input_for_decoder(input_sent, inp_lens_tensor, self.model_input_emb, self.model_enc, self.device)

            sos_embed = self.model_output_emb.forward(torch.tensor(sos_idx).unsqueeze(0).to(self.device))
            pred, hidden = self.model_dec.forward(sos_embed.unsqueeze(0), context, enc_word)
            val,ind = torch.sort(pred.squeeze(0)[0],0)
            #print(val,ind)
            top_3 = ind[-beam_size:] # top3 prediction from beamsearch
            for k in range(0,beam_size): #store 3 index hyptothesis
                hyps.append(Hypothesis(tokens=[top_3[k]], log_probs=pred.squeeze(0)[0][top_3[k]], state=hidden, attn_dists=[],p_gens=[], coverage=[] ) )
            
            results = []
            original_ans = []
            length = 0
            while length < self.output_max_len and len(results) < beam_size:
                if(length<(len(ex.best_x))):
                    original_ans.append(self.output_indexer.get_object(ex.best_x[length]))
                var = 2*beam_size
                all_hyps = []
                for h in hyps:
                    latest_token = h.latest_token
                    hidden = h.state
                    word_emb = self.model_output_emb.forward(torch.tensor(latest_token).unsqueeze(0).to(self.device))
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
                
            print(output_seq)
            joint_ans = " ".join(output_seq)
            
            print("joint answers are", joint_ans)
            print(type(joint_ans))
            
            joint_orig = " ".join(original_ans)
            print("joint answers are", joint_ans)
            print("joint original are", joint_orig)
            print(type(joint_ans))
            
            f.write(joint_ans+"\n")
            g.write(joint_orig+"\n")
        f.close()
        g.close()
        #test_derivs.append([Derivation(ex, 1.0, output_seq)])
        return test_derivs


class Summarizer_cov(object):
    
    def __init__(self, input_indexer, output_indexer, model_enc,model_dec,model_input_emb,model_output_emb,reverse_input,output_max_len, device, input_max_len):
        self.input_indexer = input_indexer
        self.output_indexer = output_indexer
        self.model_enc = model_enc
        self.model_dec = model_dec
        self.model_input_emb = model_input_emb
        self.model_output_emb = model_output_emb
        self.reverse_input = reverse_input
        self.output_max_len = output_max_len
        self.device = device
        self.input_max = input_max_len
        #raise Exception("implement me!")
        # Add any args you need here

    def decode(self, test_data):
        self.model_enc.eval()
        self.model_dec.eval()
        self.model_input_emb.eval()
        self.model_output_emb.eval()
        f= open("answers_simple_cov_decode.txt","w")
        g= open("answers_original_cov_decode.txt","w")
        sos_idx = self.output_indexer.index_of(SOS_SYMBOL)
        eos_idx = self.output_indexer.index_of(EOS_SYMBOL)
        test_derivs =[]
        #input_max_len = np.max(np.asarray([len(ex.doc) for ex in test_data]))
        all_test_input_data = make_padded_input_tensor(test_data, self.input_indexer, self.input_max, self.reverse_input)

        for idx in range(len(test_data)):
            ex = test_data[idx]
            
            input_sent = torch.from_numpy(all_test_input_data[idx]).unsqueeze(0).to(self.device)
            inp_lens_tensor = torch.from_numpy(np.array(len(all_test_input_data[idx]))).unsqueeze(0).to(self.device)
            enc_word, b, context = encode_input_for_decoder(input_sent, inp_lens_tensor, self.model_input_emb, self.model_enc, self.device)

            attention_weights =  torch.autograd.Variable(torch.FloatTensor([0]*len(all_test_input_data[idx])))
            coverage_vec = torch.autograd.Variable(torch.FloatTensor([0]*len(all_test_input_data[idx])))

            sos_embed = self.model_output_emb.forward(torch.tensor(sos_idx).unsqueeze(0).to(self.device))
            pred, hidden, coverage_vec, attention_weights  = self.model_dec.forward(sos_embed.unsqueeze(0), context, enc_word, coverage_vec, attention_weights)
            
            out_idx = torch.max(pred.squeeze(0), dim = 1)[1]
            
            length = 0
            original_ans = []
            output_seq = []
            while out_idx!=eos_idx and length < self.output_max_len:
                
                output_seq.append(self.output_indexer.get_object(out_idx.item()))
                if(length<(len(ex.best_x))):
                    original_ans.append(self.output_indexer.get_object(ex.best_x[length]))
                word_embed = self.model_output_emb.forward(torch.tensor(out_idx.item()).unsqueeze(0).to(self.device))
                pred, hidden,coverage_vec, attention_weights = self.model_dec.forward(word_embed.unsqueeze(0), hidden, enc_word, coverage_vec, attention_weights)
                out_idx = torch.max(pred.squeeze(0), dim = 1)[1]
                length += 1
            print(output_seq)
            joint_ans = " ".join(output_seq)
            
            joint_orig = " ".join(original_ans)
            print("joint answers are", joint_ans)
            print("joint original are", joint_orig)
            print(type(joint_ans))
            
            f.write(joint_ans+"\n")
            g.write(joint_orig+"\n")
            #test_derivs.append([Derivation(ex, 1.0, output_seq)])
        f.close()
        g.close()
        return test_derivs
    
    def beamsearch_decode(self, test_data):
        self.model_enc.eval()
        self.model_dec.eval()
        self.model_input_emb.eval()
        self.model_output_emb.eval()
        f= open("answers_cov_beamsearch.txt","w")
        g= open("answers_cov_original_decode.txt","w")
        beam_size = 9
        sos_idx = self.output_indexer.index_of(SOS_SYMBOL)
        eos_idx = self.output_indexer.index_of(EOS_SYMBOL)
        test_derivs =[]
        print("running beam search decode")   
        input_max_len = np.max(np.asarray([len(ex.doc) for ex in test_data]))
        all_test_input_data = make_padded_input_tensor(test_data, self.input_indexer, self.input_max, self.reverse_input)

        for idx in range(len(test_data)):#
            ex = test_data[idx]
            hyps=[]
            input_sent = torch.from_numpy(all_test_input_data[idx]).unsqueeze(0).to(self.device)
            inp_lens_tensor = torch.from_numpy(np.array(len(all_test_input_data[idx]))).unsqueeze(0).to(self.device)
            enc_word, b, context = encode_input_for_decoder(input_sent, inp_lens_tensor, self.model_input_emb, self.model_enc, self.device)

            sos_embed = self.model_output_emb.forward(torch.tensor(sos_idx).unsqueeze(0).to(self.device))
            attention_weights =  torch.autograd.Variable(torch.FloatTensor([0]*len(all_test_input_data[idx])))
            coverage_vec = torch.autograd.Variable(torch.FloatTensor([0]*len(all_test_input_data[idx])))

            pred, hidden, coverage_vec, attention_weights = self.model_dec.forward(sos_embed.unsqueeze(0), context, enc_word, coverage_vec, attention_weights)
            val,ind = torch.sort(pred.squeeze(0)[0],0)
            #print(val,ind)
            top_3 = ind[-beam_size:] # top3 prediction from beamsearch
            for k in range(0,beam_size): #store 3 index hyptothesis
                hyps.append(Hypothesis(tokens=[top_3[k]], log_probs=pred.squeeze(0)[0][top_3[k]], state=hidden, attn_dists=[],p_gens=[], coverage=[] ) )
            
            results = []
            original_ans = []
            length = 0
            while length < self.output_max_len and len(results) < beam_size:
                if(length<(len(ex.best_x))):
                    original_ans.append(self.output_indexer.get_object(ex.best_x[length]))
                var = 2*beam_size
                all_hyps = []
                for h in hyps:
                    latest_token = h.latest_token
                    hidden = h.state
                    word_emb = self.model_output_emb.forward(torch.tensor(latest_token).unsqueeze(0).to(self.device))
                    pred, hidden, coverage_vec, attention_weights = self.model_dec.forward(word_emb.unsqueeze(0), hidden, enc_word, coverage_vec, attention_weights)
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
                
            print(output_seq)
            joint_ans = " ".join(output_seq)
            
            print("joint answers are", joint_ans)
            print(type(joint_ans))
            
            joint_orig = " ".join(original_ans)
            print("joint answers are", joint_ans)
            print("joint original are", joint_orig)
            print(type(joint_ans))
            
            f.write(joint_ans+"\n")
            g.write(joint_orig+"\n")
        f.close()
        g.close()
        #test_derivs.append([Derivation(ex, 1.0, output_seq)])
        return test_derivs


def sort_hyps(hyps):
    """#Return a list of Hypothesis objects, sorted by descending average log probability"""
    return sorted(hyps, key=lambda h: h.avg_log_prob(), reverse=True)

# Takes the given Examples and their input indexer and turns them into a numpy array by padding them out to max_len.
# Optionally reverses them.
def make_padded_input_tensor(exs, input_indexer, max_len, reverse_input):
    if reverse_input:
        return np.array(
            [[ex.doc[len(ex.doc) - 1 - i] if i < len(ex.doc) else input_indexer.index_of(PAD_SYMBOL)
              for i in range(0, max_len)]
             for ex in exs])
    else:
        return np.array([[ex.doc[i] if i < len(ex.doc) else input_indexer.index_of(PAD_SYMBOL)
                          for i in range(0, max_len)]
                         for ex in exs])

# Analogous to make_padded_input_tensor, but without the option to reverse input
def make_padded_output_tensor(exs, output_indexer, max_len):
    return np.array([[ex.best_x[i] if i < len(ex.best_x) else output_indexer.index_of(PAD_SYMBOL) for i in range(0, max_len)] for ex in exs])

def encode_input_for_decoder(x_tensor, inp_lens_tensor, model_input_emb, model_enc, device):
    input_emb = model_input_emb.forward(x_tensor)
    (enc_output_each_word, enc_context_mask, enc_final_states) = model_enc.forward(input_emb, inp_lens_tensor)
    enc_final_states_reshaped = (enc_final_states[0].unsqueeze(0), enc_final_states[1].unsqueeze(0))
    return (enc_output_each_word, enc_context_mask, enc_final_states_reshaped)


def train_model_encdec(train_data, test_data, input_indexer, output_indexer, args, device):
    # Sort in descending order by x_indexed, essential for pack_padded_sequence
    train_data.sort(key=lambda ex: len(ex.doc), reverse=True)
    test_data.sort(key=lambda ex: len(ex.doc), reverse=True)
   
    # Create indexed input
    input_max_len = np.max(np.asarray([len(ex.doc) for ex in train_data]))
    all_train_input_data = make_padded_input_tensor(train_data, input_indexer, input_max_len, args.reverse_input)
    all_test_input_data = make_padded_input_tensor(test_data, input_indexer, input_max_len, args.reverse_input)

    output_max_len = np.max(np.asarray([len(ex.best_x) for ex in train_data]))
    median = np.median(np.asarray([len(ex.best_x) for ex in train_data]))
    all_train_output_data = make_padded_output_tensor(train_data, output_indexer, output_max_len)
    all_test_output_data = make_padded_output_tensor(test_data, output_indexer, output_max_len)

    print("Train length: %i" % input_max_len, output_max_len)
    #print("Train output length: %i" % np.max(np.asarray([len(ex.x_best) for ex in train_data])))
    print("Train matrix: %s; shape = %s" % (all_train_input_data, all_train_input_data.shape))

    # Create model
    model_input_emb = EmbeddingLayer(args.input_dim, len(input_indexer), args.emb_dropout).to(device)
    model_enc = RNNEncoder(args.input_dim, args.hidden_size, args.rnn_dropout, args.bidirectional).to(device)
    model_output_emb = EmbeddingLayer(args.output_dim, len(output_indexer), args.emb_dropout).to(device)
    model_dec = LSTMAttnClassifier(args.output_dim, args.hidden_size, len(output_indexer)).to(device)
    
    print(model_enc)
    print(model_dec)
    print(model_input_emb)
    print(model_output_emb)
    
    sos_idx = output_indexer.index_of(SOS_SYMBOL)
    eos_idx = output_indexer.index_of(EOS_SYMBOL)
    criterion = torch.nn.NLLLoss().to(device)
    params = list(model_enc.parameters()) + list(model_dec.parameters()) + list(model_input_emb.parameters()) + list(model_output_emb.parameters())
    optimizer = torch.optim.Adam(params, lr=0.0001)
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
           
            input_sent = torch.from_numpy(all_train_input_data[idx]).unsqueeze(0).to(device)
            inp_lens_tensor = torch.from_numpy(np.array(len(train_data[idx].doc))).unsqueeze(0).to(device)
            enc_word, b, context = encode_input_for_decoder(input_sent, inp_lens_tensor, model_input_emb, model_enc, device)
           
            inp_idx = sos_idx
            hidden = context
            ex = train_data[idx]
            #print("The original doc is this", len(ex.doc), ex.doc)
            #print("The best_x is this",  len(ex.best_x), ex.best_x)
            #"""This is simple training, For training scheduled sampling - make this False n below True
            if True:
                for tgt_idx in train_data[idx].best_x:
                    word_embed = model_output_emb.forward(torch.tensor(inp_idx).unsqueeze(0).to(device))
                    pred, hidden = model_dec.forward(word_embed.unsqueeze(0), hidden, enc_word)
                    sent_loss +=  criterion(pred.squeeze(0), torch.LongTensor([tgt_idx]).to(device)).to(device)
                    truth_label.append(tgt_idx)
                    pred_label.append(torch.max(pred.squeeze(0), dim = 1)[1].item())
                    #print("tgt_idx", tgt_idx)
                    #print("learnt word", tgt_idx, (torch.max(pred.squeeze(0), dim = 1)[1].item()))
                    inp_idx = tgt_idx
                    if tgt_idx == eos_idx:
                        break
                
            if False: # Make true for scheduled sampling 
                use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    
                for tgt_idx in train_data[idx].best_x:
                    
                    word_embed = model_output_emb.forward(torch.tensor(inp_idx).unsqueeze(0).to(device))
                    pred, hidden = model_dec.forward(word_embed.unsqueeze(0), hidden, enc_word)
                    sent_loss +=  criterion(pred.squeeze(0), torch.LongTensor([tgt_idx])).to(device)
                    if use_teacher_forcing:
                        inp_idx = tgt_idx
                    else:
                        inp_idx = torch.max(pred.squeeze(0), dim = 1)[1].item().to(device)
                        
                    if inp_idx == eos_idx:
                        break
            count = count +1            
            sent_loss.backward()
            #print("truth label",truth_label, pred_label)
            optimizer.step()
        print("epoch", epoch, "predicted index", pred_label, "truth index", truth_label)
    print(truth_label, pred_label)
    print ("finished training")  
    
    path_to_input_emb = "./input_emb.pth" 
    path_to_output_emb = "./output_emb.pth"
    path_to_enc = "./enc.pth"
    path_to_dec = "./dec.pth"
    torch.save(model_input_emb.state_dict(), path_to_input_emb)
    torch.save(model_output_emb.state_dict(), path_to_output_emb)
    torch.save(model_enc.state_dict(), path_to_enc)
    torch.save(model_dec.state_dict(), path_to_dec)


    return Summarizer(input_indexer, output_indexer, model_enc,model_dec,model_input_emb,model_output_emb,args.reverse_input,median, device)

def train_model_encdec_cov(train_data, test_data, input_indexer, output_indexer, args, device):
    # Sort in descending order by x_indexed, essential for pack_padded_sequence
    print("cov is called")
    train_data.sort(key=lambda ex: len(ex.doc), reverse=True)
    test_data.sort(key=lambda ex: len(ex.doc), reverse=True)
   
    # Create indexed input
    input_max_len = np.max(np.asarray([len(ex.doc) for ex in train_data]))
    all_train_input_data = make_padded_input_tensor(train_data, input_indexer, input_max_len, args.reverse_input)
    all_test_input_data = make_padded_input_tensor(test_data, input_indexer, input_max_len, args.reverse_input)

    output_max_len = np.max(np.asarray([len(ex.best_x) for ex in train_data]))
    median = np.median(np.asarray([len(ex.best_x) for ex in train_data]))
    all_train_output_data = make_padded_output_tensor(train_data, output_indexer, output_max_len)
    all_test_output_data = make_padded_output_tensor(test_data, output_indexer, output_max_len)

    print("Train length: %i" % input_max_len, output_max_len)
    #print("Train output length: %i" % np.max(np.asarray([len(ex.x_best) for ex in train_data])))
    print("Train matrix: %s; shape = %s" % (all_train_input_data, all_train_input_data.shape))

    # Create model
    model_input_emb = EmbeddingLayer(args.input_dim, len(input_indexer), args.emb_dropout).to(device)
    model_enc = RNNEncoder(args.input_dim, args.hidden_size, args.rnn_dropout, args.bidirectional).to(device)
    model_output_emb = EmbeddingLayer(args.output_dim, len(output_indexer), args.emb_dropout).to(device)
    model_dec = LSTMCovClassifier(args.output_dim, args.hidden_size, len(output_indexer), input_max_len).to(device)
    
    print(model_enc)
    print(model_dec)
    print(model_input_emb)
    print(model_output_emb)
    
    sos_idx = output_indexer.index_of(SOS_SYMBOL)
    eos_idx = output_indexer.index_of(EOS_SYMBOL)
    criterion = torch.nn.NLLLoss().to(device)
    params = list(model_enc.parameters()) + list(model_dec.parameters()) + list(model_input_emb.parameters()) + list(model_output_emb.parameters())
    optimizer = torch.optim.Adam(params, lr=0.0001)
    # Loop over epochs, loop over examples, given some indexed words, call encode_input_for_decoder, then call your
    # decoder, accumulate losses, update parameters
    model_enc.train()
    model_dec.train()
    model_input_emb.train()
    model_output_emb.train()
    count = 0 
    for epoch in range(0,60):#20
        perm0 =np.arange(len(train_data))
        
        random.shuffle(perm0)
        for idx in perm0:
            
            truth_label = []
            pred_label = []
            
            teacher_forcing_ratio = (0.999)**(count)
            optimizer.zero_grad()
            
            sent_loss = 0
           
            input_sent = torch.from_numpy(all_train_input_data[idx]).unsqueeze(0).to(device)
            inp_lens_tensor = torch.from_numpy(np.array(len(all_train_input_data[idx]))).unsqueeze(0).to(device)
            enc_word, b, context = encode_input_for_decoder(input_sent, inp_lens_tensor, model_input_emb, model_enc, device)
           
            #print(enc_word.size())
            #print(b.size())
            inp_idx = sos_idx
            hidden = context
            ex = train_data[idx]

            attention_weights =  torch.autograd.Variable(torch.FloatTensor([0]*len(all_train_input_data[idx])))
            coverage_vec = torch.autograd.Variable(torch.FloatTensor([0]*len(all_train_input_data[idx])))

            if True:
                for tgt_idx in train_data[idx].best_x:
                    word_embed = model_output_emb.forward(torch.tensor(inp_idx).unsqueeze(0).to(device))
                    pred, hidden, coverage_vec, attention_weights = model_dec.forward(word_embed.unsqueeze(0), hidden, enc_word, coverage_vec, attention_weights)
                    #print("min value",torch.sum(torch.min(coverage_vec,attention_weights)).item())
                    sent_loss +=  criterion(pred.squeeze(0), torch.LongTensor([tgt_idx]).to(device)).to(device) + torch.sum(torch.min(coverage_vec,attention_weights))
                    truth_label.append(tgt_idx)
                    pred_label.append(torch.max(pred.squeeze(0), dim = 1)[1].item())
                    #print("tgt_idx", tgt_idx)
                    #print("learnt word", tgt_idx, (torch.max(pred.squeeze(0), dim = 1)[1].item()))
                    inp_idx = tgt_idx
                    if tgt_idx == eos_idx:
                        break
                
            if False: # Make true for scheduled sampling 
                use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    
                for tgt_idx in train_data[idx].best_x:
                    
                    word_embed = model_output_emb.forward(torch.tensor(inp_idx).unsqueeze(0).to(device))
                    pred, hidden, coverage_vec, attention_weights = model_dec.forward(word_embed.unsqueeze(0), hidden, enc_word, coverage_vec, attention_weights)
                    sent_loss +=  criterion(pred.squeeze(0), torch.LongTensor([tgt_idx])).to(device) 
                    if use_teacher_forcing:
                        inp_idx = tgt_idx
                    else:
                        inp_idx = torch.max(pred.squeeze(0), dim = 1)[1].item().to(device)
                        
                    if inp_idx == eos_idx:
                        break
            count = count +1            
            sent_loss.backward()
            #print("truth label",truth_label, pred_label)
            optimizer.step()
        print("epoch", epoch, "predicted index", pred_label, "truth index", truth_label)
    print(truth_label, pred_label)
    print ("finished training")  
    
    path_to_input_emb = "./input_emb_cov.pth" 
    path_to_output_emb = "./output_emb_cov.pth"
    path_to_enc = "./enc_cov.pth"
    path_to_dec = "./dec_cov.pth"
    torch.save(model_input_emb.state_dict(), path_to_input_emb)
    torch.save(model_output_emb.state_dict(), path_to_output_emb)
    torch.save(model_enc.state_dict(), path_to_enc)
    torch.save(model_dec.state_dict(), path_to_dec)
    
    
    #vgg = models.vgg16()
    
    #summary(model.cuda(), (INPUT_SHAPE))
    print("input max len",input_max_len)
    return Summarizer_cov(input_indexer, output_indexer, model_enc,model_dec,model_input_emb,model_output_emb,args.reverse_input,median, device, input_max_len)

def evaluate(decoder, test_exs):
    predicted_exs = decoder.decode(test_exs)
    predicted_exs = decoder.beamsearch_decode(test_exs)

def render_ratio(numer, denom):
    return "%i / %i = %.3f" % (numer, denom, float(numer)/denom)


if __name__ == '__main__':
    args = _parse_args()
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    word_vectors = read_word_embeddings(args.word_vecs_path)
    ID_dict, train_ques_list, dev_ques_list, num_classes =  read_and_index_question_examples(args.para_ques_url, word_vectors.word_indexer)
    
    #print(train_ques_list[100])
    #answer_exs =  read_and_index_answer_examples(args.answer_file, ID_dict, word_vectors.word_indexer)
    answer_exs, indexer =  read_and_index_answer_examples_new(args.answer_file, ID_dict, word_vectors.word_indexer)
    print(len(indexer))
    print(len(answer_exs))
    decoder =  train_model_encdec_cov(answer_exs, answer_exs[1:10], indexer, indexer, args, device)

    #decoder =  train_model_encdec(answer_exs, answer_exs[1:10], word_vectors.word_indexer, word_vectors.word_indexer, args, device)
    
    evaluate(decoder, answer_exs[1:10])
    #evaluate(dev_data_indexed, decoder) #to evalaute beam search use this evaluate_bs(dev_data_indexed, decoder)
    #print("=======FINAL EVALUATION ON BLIND TEST=======")
    #evaluate(test_data_indexed, decoder, print_output=False, outfile="geo_test_output.tsv")
  
"""   
from pythonrouge.pythonrouge import Pythonrouge

# system summary(predict) & reference summary
summary = [[" Tokyo is the one of the biggest city in the world."]]
reference = [[["The capital of Japan, Tokyo, is the center of Japanese economy."]]]

# initialize setting of ROUGE to eval ROUGE-1, 2, SU4
# if you evaluate ROUGE by sentence list as above, set summary_file_exist=False
# if recall_only=True, you can get recall scores of ROUGE
rouge = Pythonrouge(summary_file_exist=False,
                    summary=summary, reference=reference,
                    n_gram=2, ROUGE_SU4=True, ROUGE_L=False,
                    recall_only=True, stemming=True, stopwords=True,
                    word_level=True, length_limit=True, length=50,
                    use_cf=False, cf=95, scoring_formula='average',
                    resampling=True, samples=1000, favor=True, p=0.5)
score = rouge.calc_score()
print(score)
    


r = Rouge155('/Users/shivi/.pyrouge')
#r = Rouge155()
r.system_dir = 'path/to/system_summaries'
r.model_dir = 'path/to/model_summaries'
r.system_filename_pattern = 'some_name.(\d+).txt'
r.model_filename_pattern = 'some_name.[A-Z].#ID#.txt'

output = r.convert_and_evaluate()
print(output)
output_dict = r.output_to_dict(output)

"""