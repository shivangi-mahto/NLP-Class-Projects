#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 18:18:28 2018

@author: shivi
"""

#results

#after gradient batch size =1 epoch 1 Labeled F1: 83.79, precision: 4950/5872 = 84.30, recall: 4950/5943 = 83.29

#batch szie =1 
            #for epoch 2 with bs 1 - Labeled F1: 86.93, precision: 5111/5816 = 87.88, recall: 5111/5943 = 86.00
            # for 3 epoch Labeled F1: 87.64, precision: 5165/5844 = 88.38, recall: 5165/5943 = 86.91
            #for epoch 5 Labeled F1: 88.34, precision: 5202/5834 = 89.17, recall: 5202/5943 = 87.53
            #for 8 Labeled F1: 88.28, precision: 5172/5774 = 89.57, recall: 5172/5943 = 87.03
#SVM for loop 1
#Labeled F1: 78.53, precision: 4617/5816 = 79.38, recall: 4617/5943 = 77.69          
#for 2 Labeled F1: 80.93, precision: 4802/5924 = 81.06, recall: 4802/5943 = 80.80
#for 3 Labeled F1: 81.95, precision: 4855/5906 = 82.20, recall: 4855/5943 = 81.69
#for 5 epochs #Labeled F1: 84.12, precision: 4954/5836 = 84.89, recall: 4954/5943 = 83.36
#for 8 Labeled F1: 84.07, precision: 4991/5931 = 84.15, recall: 4991/5943 = 83.98          
#    #feat_tran_cache = [[[] for k in range(0, len(tag_indexer))] for j in range(0, len(tag_indexer))]

#For 11600 smv with prob Labeled F1: 85.83, precision: 5066/5862 = 86.42, recall: 5066/5943 = 85.24
# SVM with transition priobs
#for loop 1 Labeled F1: 83.06, precision: 4926/5919 = 83.22, recall: 4926/5943 = 82.89
#for loop 2 Labeled F1: 85.43, precision: 5011/5788 = 86.58, recall: 5011/5943 = 84.32
#for loop 5 Labeled F1: 87.57, precision: 5169/5862 = 88.18, recall: 5169/5943 = 86.98
#for loop 8 Labeled F1: 87.87, precision: 5189/5868 = 88.43, recall: 5189/5943 = 87.31
#SVM for SGD:
#    Labeled F1: 79.80, precision: 4773/6020 = 79.29, recall: 4773/5943 = 80.31
#def extract_transition_features(prev_tag, tag, feature_indexer, add_to_indexer):
#    feats = []
    
#   maybe_add_feature(feats, feature_indexer, add_to_indexer, prev_tag + "-" + tag)

#   return np.asarray(feats, dtype=int)

                    #Marg_prob[i][s] = (np.exp((alpha[i][s]) + (beta[i][s]) )) / (np.sum((alpha[i])+(beta[i])))
                        
                #print ("i, should be same for all", i, np.log(np.sum(np.exp(alpha[i]+beta[i]))) ) # should be same for all i
                
#for batch size in len(sentence)
                #for sentence_idx in range(0, 2):
    #for 28 Labeled F1: 87.92, precision: 5168/5813 = 88.90, recall: 5168/5943 = 86.96 Data reading and training took 43009.222785 seconds
    #Running on test
    # for 22 Labeled F1: 87.86, precision: 5160/5803 = 88.92, recall: 5160/5943 = 86.82
    #for 15 epoch Labeled F1: 87.52, precision: 5138/5798 = 88.62, recall: 5138/5943 = 86.45 Data reading and training took 6384.325949 seconds
    #for 18 Labeled F1: 87.66, precision: 5148/5803 = 88.71, recall: 5148/5943 = 86.62
    ##5 LEADS TO 85.45F1 WITH PRECISION 87.02 AND RECALL 83.93 # for 8 F1: 86.44, precision: 5060/5764 = 87.79, recall: 5060/5943 = 85.14 #12 leads Labeled F1: 87.07, precision: 5110/5795 = 88.18, recall: 5110/5943 = 85.98
                      #temp_score[prev_tag] = score[i-1][prev_tag] + self.transition_log_probs[prev_tag,s] + self.emission_log_probs[s,self.word_indexer.index_of(word_i)] 
#                  #temp_score[prev_tag] = score[i-1][prev_tag] + self.transition_log_probs[prev_tag,s] + self.emission_log_probs[s,self.word_indexer.index_of(word_i)] 
