#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 18:29:47 2018

@author: shivi
"""
import argparse
import sys
from model import *
from data import *

def _parse_args():
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--model', type=str, default='FFNN', help='model to run (FF or FANCY)')
    parser.add_argument('--word_vecs_path', type=str, default='data/glove.6B.300d-relativized.txt', help='path to word vectors file')
    parser.add_argument('--train_path', type=str, default='quora_duplicate_questions.tsv', help='path to train set (you should not need to modify)')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True    
    word_vectors = read_word_embeddings(args.word_vecs_path)
    quora_exs = read_and_index_sentiment_examples(args.train_path, word_vectors.word_indexer)
    print(len(quora_exs[0]), len(quora_exs[1]))

    #if args.model == "FFNN":
    test_exs_predicted = train_model(quora_exs, word_vectors, device)
    
    #print(len(quora_exs))
    #print(quora_exs[1000])
    
