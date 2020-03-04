#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 23:29:20 2018

@author: shivi
"""

# classifier_main.py

import argparse
import sys
import time
from nerdata import *
from utils import *
from optimizers import *
import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier

def sigmoid(x):
  return 1 / (1 + np.exp(-x))
# Command-line arguments to the system -- you can extend these if you want, but you shouldn't need to modify any of them
def _parse_args():
    parser = argparse.ArgumentParser(description='trainer.py')
    parser.add_argument('--model', type=str, default='BAD', help='model to run (BAD, CLASSIFIER)')
    parser.add_argument('--train_path', type=str, default='data/eng.train', help='path to train set (you should not need to modify)')
    parser.add_argument('--dev_path', type=str, default='data/eng.testa', help='path to dev set (you should not need to modify)')
    parser.add_argument('--blind_test_path', type=str, default='data/eng.testb.blind', help='path to dev set (you should not need to modify)')
    parser.add_argument('--test_output_path', type=str, default='eng.testb.out', help='output path for test predictions')
    parser.add_argument('--no_run_on_test', dest='run_on_test', default=True, action='store_false', help='skip printing output on the test set')
    args = parser.parse_args()
    return args


# Wrapper for an example of the person binary classification task.
# tokens: list of string words
# labels: list of (0, 1) where 0 is non-name, 1 is name
class PersonExample(object):
    def __init__(self, tokens, labels):
        self.tokens = tokens
        self.labels = labels

    def __len__(self):
        return len(self.tokens)


# Changes NER-style chunk examples into binary classification examples.
def transform_for_classification(ner_exs):
    for labeled_sent in ner_exs:
        tags = bio_tags_from_chunks(labeled_sent.chunks, len(labeled_sent))
        labels = [1 if tag.endswith("PER") else 0 for tag in tags]
        yield PersonExample([tok.word for tok in labeled_sent.tokens], labels)


# Person classifier that takes counts of how often a word was observed to be the positive and negative class
# in training, and classifies as positive any tokens which are observed to be positive more than negative.
# Unknown tokens or ties default to negative.
class CountBasedPersonClassifier(object):
    def __init__(self, pos_counts, neg_counts):
        self.pos_counts = pos_counts
        self.neg_counts = neg_counts

    def predict(self, tokens, idx):
        if self.pos_counts.get_count(tokens[idx]) > self.neg_counts.get_count(tokens[idx]):
            return 1
        else:
            return 0


# "Trains" the count-based person classifier by collecting counts over the given examples.
def train_count_based_binary_classifier(ner_exs):
    pos_counts = Counter()
    neg_counts = Counter()
    for ex in ner_exs:
        for idx in range(0, len(ex)):
            if ex.labels[idx] == 1:
                pos_counts.increment_count(ex.tokens[idx], 1.0)
            else:
                neg_counts.increment_count(ex.tokens[idx], 1.0)
    return CountBasedPersonClassifier(pos_counts, neg_counts)

#------------------------------------------------------------------------------
# "Real" classifier that takes a weight vector
class PersonClassifier(object):
    def __init__(self, weights, counter, indexer,indexer_2gram):
        self.weights = weights
        self.counter = counter
        self.indexer = indexer
        self.indexer_2gram = indexer_2gram
    # Makes a prediction for token at position idx in the given PersonExample
    def predict(self, tokens, idx):
        word = tokens[idx]
        #word_feat = np.array([int(word.istitle()), int(word.islower()), int(word.isupper()), len(word), int(word.isdigit()),  int(word.isalpha())])
        countclf = self.counter
        count_prev = 0.0
        count_next = 0.0
        word_prev = 'Null'
        word_next = 'Null'
        if (idx-1>=0):
            count_prev = countclf.predict(tokens,idx-1)
            word_prev = tokens[idx-1]
        if (idx+1<len(tokens)):
            count_next = countclf.predict(tokens,idx+1)
            word_next = tokens[idx+1]
        count_curr = countclf.predict(tokens,idx)
        word_feat = np.concatenate((get_word_feature(word),[count_prev,count_curr, count_next]),axis=None)

        len_bow = len(self.indexer.objs_to_ints)
        index_bow = self.indexer.index_of(word)
        index_2gr = self.indexer_2gram.index_of(word+" "+ word_next)
        #index_prev = self.indexer.index_of(word_prev)
        #index_next = self.indexer.index_of(word_next)
        #scores = np.dot(word_feat, self.weights)
        score1 = np.dot(word_feat, self.weights[0:9])
        score2 = 1*self.weights[9+int(index_bow)]
        score3 = 1*self.weights[9+len_bow+int(index_2gr)]
        scores = score1+score2+score3
        #scores = self.weights[int(index)] 
        #scores = 1*self.weights[int(index_prev)] + 1*self.weights[len_bow+int(index)] + 1*self.weights[2*len_bow+int(index_next)]
        #scores = 1*self.weights[int(index)] + 1*self.weights[len_bow+int(index_next)] 
        #print((scores[0]))
        return round(1 / (1 + math.exp(-scores[0])))

class RandomForest(RandomForestClassifier):
    
    def __init__(self, n_estimators,countclf):
        RandomForestClassifier.__init__(self, n_estimators = n_estimators)
        self.countclf = countclf
    # Makes a prediction for token at position idx in the given PesonExample
    def predictor(self, tokens, idx):
        word = tokens[idx]
        #word_feat = np.array([int(word.istitle()), int(word.islower()), int(word.isupper()), len(word), int(word.isdigit()),  int(word.isalpha())])
        countclf = self.countclf
        count_prev = 0.0
        count_next = 0.0
        word_prev = 'Null'
        word_next = 'Null'
        if (idx-1>=0):
            count_prev = countclf.predict(tokens,idx-1)
            word_prev = tokens[idx-1]
        if (idx+1<len(tokens)):
            count_next = countclf.predict(tokens,idx+1)
            word_next = tokens[idx+1]
        count_curr = countclf.predict(tokens,idx)
        word_feat = np.concatenate((get_word_feature(word_prev), get_word_feature(word), get_word_feature(word_next),[count_prev,count_curr, count_next]),axis=None)
       
        pred = self.predict([word_feat])
        return pred[0]

def get_word_feature(word):
    return int(word.istitle()), int(word.islower()), int(word.isupper()), len(word), int(word.isdigit()), int(word.isalpha())
    
def feature_extractor(ner_exs):
    countclf = train_count_based_binary_classifier(ner_exs)
    indexer = Indexer() # collect bag of individual words
    indexer_2gram = Indexer() 
    for ex in ner_exs:
        for idx in range(0, len(ex)):
            indexer.get_index(ex.tokens[idx])
            if idx+1<len(ex):
                indexer_2gram.get_index(ex.tokens[idx]+" "+ ex.tokens[idx+1])
    len_indexer = len(indexer.objs_to_ints)
    len_2gram = len(indexer_2gram.objs_to_ints)


    #feat = np.empty((1,21+3))
    feat = np.empty((1,11))
    ##Sprint(len_indexer)
    labels = np.empty((1,1))
    count_prev = 0.0
    count_next = 0.0
    word_prev = 'Null'
    word_next = 'Null'
    for ex in ner_exs:
        feat_ex = np.empty((len(ex),9))
        #feat_ex = np.empty((len(ex),1))
        context_feat = np.empty((len(ex),2),dtype=int)
        #context_feat = np.empty((len(ex),11),dtype=int)
        concat_feat = np.empty((len(ex),11))
        #print("concat shape",concat_feat.shape)
        true_labels = np.empty((len(ex),1),dtype=int)
        indi_prev=indexer.index_of(word_prev)
        indi_next=indexer.index_of(word_next)
        #indi_next=np.zeros(len_indexer, dtype=int)
        #indi_word=np.zeros(len_indexer, dtype=int)
        
        for idx in range(0, len(ex)):
            count_curr = countclf.predict(ex.tokens,idx)
            if (idx-1>=0):
                count_prev = countclf.predict(ex.tokens,idx-1)
                word_prev = ex.tokens[idx-1]
                indi_prev = indexer.index_of(ex.tokens[idx-1])
            if (idx+1<len(ex)):
                count_next = countclf.predict(ex.tokens,idx+1)                
                word_next = ex.tokens[idx+1]
                indi_next = indexer.index_of(ex.tokens[idx+1])
            word = ex.tokens[idx]
            #print (get_word_featue(word))
            #feat_ex[idx] = np.concatenate((get_word_feature(word_prev), get_word_feature(word), get_word_feature(word_next),[count_prev,count_curr, count_next]),axis=None)
            feat_ex[idx] = np.concatenate((get_word_feature(word),[count_prev,count_curr, count_next]),axis=None)

            true_labels[idx] = ex.labels[idx]
            indi_word = indexer.index_of(word)
            indi_2gram = indexer_2gram.get_index(word+" "+ word_next)
            #context_feat[idx] = np.concatenate((indi_word,indi_next),axis=None)
            #context_feat[idx] = np.concatenate((indi_prev,indi_word,indi_next),axis=None)
            context_feat[idx] = np.concatenate((indi_word,indi_2gram),axis=None)
            #print(word, context_feat[idx])
            #print(context_feat.shape)
            #print(feat_ex.shape)
            concat_feat[idx] = np.concatenate(( feat_ex[idx],context_feat[idx]), axis = None)
            
        #print(concat_feat.shape)
        #feat = np.append(feat,feat_ex, axis=0)
        #feat = np.append(feat,concat_feat, axis=0)
        feat = np.append(feat,concat_feat, axis=0)      
        #print(feat)
        labels = np.append(labels,true_labels, axis=0)
        #print(feat.shape, labels.shape)
        
    return (countclf, indexer, indexer_2gram, feat, labels)

def train_classifier(ner_exs):
    countclf,indexer, indexer_2gram, __feat , __label  = feature_extractor(ner_exs)
    #print(__feat.shape, __label.shape)
    lr = 0.001
    
    #weights = np.zeros((__feat.shape[1],1))
    len_bow = len(indexer.objs_to_ints)
    len_2gram = len(indexer_2gram.objs_to_ints)
    weights = np.zeros((9+len_bow+len_2gram,1)) #1
    SGD = SGDOptimizer(weights, lr)
    AdaGrad=L1RegularizedAdagradTrainer(weights)
    Unreg_AdaGrad = UnregularizedAdagradTrainer(weights)
   
    for epoch in range(0,100):
        perm = np.random.permutation(len(__label))
        label = __label[perm,:]
        feat  = __feat[perm,:]
        data_size = feat.shape[0]
        batch_size = 256
        itr_count = math.floor(data_size/batch_size)
        #print (itr_count)
        for itr in range(0,itr_count):
            
            feat_bs = feat[itr*batch_size:(itr+1)*batch_size,:]
            feat_word = feat_bs[:,0:9]
            cont_1 = feat_bs[:,9]
            cont_2 = feat_bs[:,10]
            
            label_bs = label[itr*batch_size:(itr+1)*batch_size,:]
            #scores = np.dot(feat_bs, weights)
            #print("")
            scores = np.empty((feat_bs.shape[0]))
            for i in range(feat_bs.shape[0]):
                #index_bow = feat_bs[i][0]
                index_bow = cont_1[i]
                index_2gr = cont_2[i]
                #index_prev  = feat_bs[i][0]
                #index_next = feat_bs[i][1]
                #scores[i] = 1*weights[int(index)]      
                #scores[i] = 1*weights[int(index_prev)] + 1*weights[len_bow+int(index)] + 1*weights[2*len_bow+int(index_next)]
                score1 = np.dot(feat_word[i,:], weights[0:9])
                score2 = 1*weights[9+int(index_bow)]
                score3 = 1*weights[9+len_bow+int(index_2gr)]
                scores[i] = score1+score2+score3
                #scores[i] = 1*weights[int(index_bow)] #+ 1*weights[1*len_bow+int(index_next)]

            predictions = sigmoid(scores)
            pred_err = np.ravel(label_bs) - predictions
            #print("pred" ,pred_err)
            
            grad=Counter()
            
            #for i in range(0,feat_bs.T.shape[0]):
            #    grad.increment_count(i, sum( [feat_bs.T[i][j]*pred_err[j] for j in range(len(pred_err))] ) )
            
            #for i in range(0,len(weights)):
            #    grad.increment_count(i, sum( [feat_bs.T[i][j]*pred_err[j] for j in range(len(pred_err))] ) )
            
            for i in range(0,feat_word.T.shape[0]):
                grad.increment_count(i, sum( [feat_bs.T[i][j]*pred_err[j] for j in range(len(pred_err))] ) )

            for i in range(0,feat_bs.shape[0]):
                #index = feat_bs[i][0]
                index_bow = cont_1[i]
                index_2gr = cont_2[i]
                #index_prev  = feat_bs[i][0]
                #index_next = feat_bs[i][1]
                grad.increment_count((9+int(index_bow)), pred_err[i]) 
                grad.increment_count((9+len_bow+int(index_2gr)), pred_err[i]) 
                #grad.increment_count(len_bow+int(index_next), pred_err[i]) 
                #grad.increment_count(int(index_prev), pred_err[i]) 
                #grad.increment_count(int(len_bow+index), pred_err[i])
                #grad.increment_count(int(2*len_bow+index_next), pred_err[i])
                #print(grad.get_count(int(feat_bs[i][0])))
            if True:
                SGD.apply_gradient_update(grad,batch_size)
                weights = SGD.get_final_weights()
            
            # Simple SGD implementation
            # gradient = np.dot(feat_bs.T, pred_err)
            # weights += lr * gradient
            
            #ADAgrad
            
            if False:
                AdaGrad.apply_gradient_update(grad,batch_size)
                weights = AdaGrad.get_final_weights()
            if False:
                Unreg_AdaGrad.apply_gradient_update(grad,batch_size)
                weights = Unreg_AdaGrad.get_final_weights()
            
            counter = countclf
    
    return PersonClassifier(weights,counter,indexer,indexer_2gram)

def train_RandomForest(ner_exs):
    
    countclf, feat , label  = feature_extractor(ner_exs)
    RF = RandomForest(n_estimators = 100, countclf=countclf)
    RF.fit(feat,np.ravel(label))
       
    return RF

def evaluate_classifier(exs, classifier):
    num_correct = 0
    num_pos_correct = 0
    num_pred = 0
    num_gold = 0
    num_total = 0
    for ex in exs:
        for idx in range(0, len(ex)):
            prediction = classifier.predict(ex.tokens, idx) #ex.tokens, idx
            if prediction == ex.labels[idx]:
                num_correct += 1
            if prediction == 1:
                num_pred += 1
            if ex.labels[idx] == 1:
                num_gold += 1
            if prediction == 1 and ex.labels[idx] == 1:
                num_pos_correct += 1
            num_total += 1
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
    prec = float(num_pos_correct) / num_pred if num_pred > 0 else 0.0
    rec = float(num_pos_correct) / num_gold if num_gold > 0 else 0.0
    f1 = 2 * prec * rec/(prec + rec) if prec > 0 and rec > 0 else 0.0
    print("Precision: %i / %i = %f" % (num_pos_correct, num_pred, prec))
    print("Recall: %i / %i = %f" % (num_pos_correct, num_gold, rec))
    print("F1: %f" % f1)


# Runs prediction on exs and writes the outputs to outfile, one token per line
def predict_write_output_to_file(exs, classifier, outfile):
    f = open(outfile, 'w')
    for ex in exs:
        for idx in range(0, len(ex)):
            prediction = classifier.predict(ex.tokens, idx)
            #print(ex.tokens)
            f.write(ex.tokens[idx] + " " + repr(int(prediction)) + "\n")
        f.write("\n")
    f.close()

if __name__ == '__main__':
    start_time = time.time()
    args = _parse_args()
    print(args)
    # Load the training and test data
    train_class_exs = list(transform_for_classification(read_data(args.train_path)))
    dev_class_exs = list(transform_for_classification(read_data(args.dev_path)))
    # Train the model
    if args.model == "BAD":
        classifier = train_count_based_binary_classifier(train_class_exs)
    elif args.model == "CLF":
        classifier = train_RandomForest(train_class_exs)
    else:
        classifier = train_classifier(train_class_exs)
    print("Data reading and training took %f seconds" % (time.time() - start_time))
    # Evaluate on training, development, and test data
    print("===Train accuracy===")
    evaluate_classifier(train_class_exs, classifier)
    print("===Dev accuracy===")
    evaluate_classifier(dev_class_exs, classifier)
    if args.run_on_test:
        print("Running on test")
        test_exs = list(transform_for_classification(read_data(args.blind_test_path)))
        predict_write_output_to_file(test_exs, classifier, args.test_output_path)
        print("Wrote predictions on %i labeled sentences to %s" % (len(test_exs), args.test_output_path))

#for ex in ner_exs:
#            feat    = np.empty((len(ex),21))
#            true_labels = np.empty((len(ex),1))
#            count_prev = 0.0
#            count_next = 0.0
#            word_prev = 'Null'
#            word_next = 'Null'
#            for idx in range(0,len(ex)):
#                #count_pred = countclf(ex.tokens,idx)
#                count_curr = countclf.predict(ex.tokens,idx)
#                if (idx-1>=0):
#                    count_prev = countclf.predict(ex.tokens,idx-1)
#                    word_prev = ex.tokens[idx-1]
#                if (idx+1<len(ex)):
#                    count_next = countclf.predict(ex.tokens,idx+1)                
#                    word_next = ex.tokens[idx+1]
#                word = ex.tokens[idx]
#                #print (get_word_featue(word))
#                feat[idx] = np.concatenate((get_word_feature(word_prev), get_word_feature(word), get_word_feature(word_next),[count_prev,count_curr, count_next]),axis=None)
#                true_labels[idx] = ex.labels[idx]
            
            #clf.fit(feat,np.ravel(true_labels))
             # increment by one so next  will add 1 tree
                #weights += lr * gradient
                #ADAgrad
                #AdaGrad.apply_gradient_update(grad,len(ex))
                #weights = AdaGrad.get_final_weights()
                #Unreg_AdaGrad.apply_gradient_update(grad,len(ex))
                #weights = Unreg_AdaGrad.get_final_weights()                
        
        #raise Exception("Implement me!")
#def feature_extractor(ner_exs):
#    indexer = Indexer() # collect bag of individual words
#    for ex in ner_exs:
#        for idx in range(0, len(ex)):
#            indexer.get_index(ex.tokens[idx])
#    len_indexer= len(indexer.objs_to_ints)
#    return indexer
#    for ex in ner_exs:
#        for idx in range(0, len(ex)):   
#            word = ex.tokens[idx]
#            print(word)
#            print (type(word))
#            print (word.istitle())
#            #features of the word itself like is has all upper case, lower case or length of the word
#            word_feat = np.array([word.istitle(), word.islower(), word.isupper(), len(word), word.isdigit(),  word.isalpha()])
#            
#            #find the indicator of the current word in the bag of words
#            indi_word = np.zeros(len_indexer, dtype=int)
#            indi_word[indexer.index_of(word)] = 1
#            #find the indicator of the previous word in the bag of words
#            indi_prev=np.zeros(len_indexer, dtype=int)
#            if(idx-1>=0): 
#                indi_prev[indexer.index_of(ex.tokens[idx-1])] = 1 
#            #find the indicator of the next word in the bag of words
#            indi_next=np.zeros(len_indexer, dtype=int)
#            if(idx+1<len(ex)):
#                indi_next[indexer.index_of(ex.tokens[idx+1])] = 1           
#            context_feat = np.concatenate( (indi_prev, np.concatenate((indi_word,indi_next),axis=None)),axis=None)
#            
#            print (len_indexer)
#            print (np.count_nonzero(feature))
#            
#    return indexer
#    
