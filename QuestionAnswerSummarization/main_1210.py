#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 18:29:47 2018

@author: shivi
"""
import argparse
import sys
from model_aquila import *
from data import *
#from train_script import *

def _parse_args():
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--model', type=str, default='FFNN', help='model to run (FF or FANCY)')
    parser.add_argument('--word_vecs_path', type=str, default='data/glove.6B.300d-relativized.txt', help='path to word vectors file')
    parser.add_argument('--train_path', type=str, default='quora_duplicate_questions.tsv', help='path to train set (you should not need to modify)')
    parser.add_argument('--para_ques_url', type=str, default='https://trec.nist.gov/data/qa/2017_LiveQA/questionParaphrases.xml')
    parser.add_argument('--answer_file', type=str, default='./Answer_file.txt')
    parser.add_argument('--no_reverse_input', dest='reverse_input', default=True, action='store_false', help='disable_input_reversal')
    parser.add_argument('--emb_dropout', type=float, default=0, help='input dropout rate')
    parser.add_argument('--input_dim', type=int, default=300, help='input vector dimensionality')
    parser.add_argument('--output_dim', type=int, default=300, help='output vector dimensionality')
    parser.add_argument('--hidden_size', type=int, default=200, help='hidden state dimensionality')
    parser.add_argument('--rnn_dropout', type=float, default=0, help='dropout rate internal to encoder RNN')
    parser.add_argument('--no_bidirectional', dest='bidirectional', default=True, action='store_false', help='bidirectional LSTM')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    word_vectors = read_word_embeddings(args.word_vecs_path)
    #quora_exs = read_and_index_sentiment_examples(args.train_path, word_vectors.word_indexer)
    ID_dict, question_exs =  read_and_index_question_examples(args.para_ques_url, word_vectors.word_indexer)
    #answer_exs =  read_and_index_answer_examples(args.answer_file, ID_dict, word_vectors.word_indexer)
    print(len(word_vectors.word_indexer))
    classifier = train_classifier(question_exs,device)
    #decoder = train_model_encdec(answer_exs, answer_exs, word_vectors.word_indexer, args, device)
    #print(answer_exs)
    """
    
    print(len(question_exs))

    #if args.model == "FFNN":
    #test_exs_predicted = train_model(quora_exs, word_vectors)
    
    #if args.model == "Classifier"
    test_exs_predicted = train_classifier(question_exs,device) 
    #print(len(quora_exs))
    #print(quora_exs[1000])
    
    """
