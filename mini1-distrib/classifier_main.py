# classifier_main.py
## Shivangi Mahto ###
import argparse
import sys
import time
from nerdata import *
from utils import *
from optimizers import *
import numpy as np
import math
import string

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
    def __init__(self, weights, indexer):
        self.weights = weights
        self.indexer = indexer
        
    # Makes a prediction for token at position idx in the given PersonExample
    def predict(self, tokens, idx):
        feat = feature_extractor(tokens, idx)
        scores = 0
        for  i in range(0,len(feat)):
            scores = scores + self.weights[(i)*len(self.indexer)+self.indexer.index_of(feat[i])]       
        return round(1 / (1 + math.exp(-scores)))


def get_word_feature(word):
    return int(word.istitle()), int(word.islower()), int(word.isupper()), len(word), int(word.isdigit()), int(word.isalpha())
    


def feature_extractor(tokens, index):
    
    tokens = [('[START2]', '[START2]'), ('[START1]', '[START1]')] + list(tokens) + [('[END1]', '[END1]'), ('[END2]', '[END2]')]
   
    index += 2
    word = tokens[index]
    prevword = tokens[index - 1]
    prevprevword = tokens[index - 2]
    nextword = tokens[index + 1]
    nextnextword = tokens[index + 2]
    is_dash = '-' in word
    is_dot = '.' in word
    threegram = [word[i:i+3] for i in range(0,6)]#len(word)-3+1 just taking for first 8 letters
    twogram = [word[i:i+2] for i in range(0,9)] #len(word)-2+1
    ascii_all = all([True for c in word if c in string.ascii_lowercase])
    allcaps = word == word.capitalize()
    capitalized = word[0] in string.ascii_uppercase
    prevcapitalized = prevword[0] in string.ascii_uppercase
    nextcapitalized = prevword[0] in string.ascii_uppercase
    
    return [word,ascii_all,threegram[0],threegram[1],threegram[2],threegram[3],threegram[4],threegram[5],word.istitle(),word.islower(),word.isupper(),nextword,len(word),word.isdigit(),nextnextword,prevword,word.isalpha(),prevprevword,is_dash,is_dot,allcaps,capitalized,prevcapitalized,nextcapitalized]
    #for context dependent return following
    #return [word,nextword,nextnextword,prevprevword,prevword]
    #for context + 3-gram
    #return [word,nextword,nextnextword,prevprevword,prevword,threegram[0],threegram[1],threegram[2],threegram[3],threegram[4],threegram[5]
    #for baseline
    #return [word]
def train_classifier(ner_exs):
    indexer = Indexer()
    lr = 0.1
    for ex in ner_exs:
        for idx in range(0, len(ex)):
            feat = feature_extractor(ex.tokens, idx)
            for i in range(0,len(feat)):
                indexer.get_index((feat[i]))
            
    weights = np.zeros((len(feat)*len(indexer)))
    SGD = SGDOptimizer(weights, lr)
    AdaGrad=L1RegularizedAdagradTrainer(weights)
    Unreg_AdaGrad = UnregularizedAdagradTrainer(weights)
    for epoch in range(50):
        for ex in ner_exs:
            for idx in range(0, len(ex)):
                feat = feature_extractor(ex.tokens, idx)
                label = ex.labels[idx]
                
                # compouting wTx by selecting indices of long weight vector
                scores =0
                for  i in range(0,len(feat)):
                    scores = scores + weights[(i)*len(indexer)+indexer.index_of(feat[i])]
            
                #prediction of logistic regression
                predictions = sigmoid(scores)
                #predication error 
                pred_err = label - predictions

                #Computing Gradient
                
                grad=Counter()
               
                for i in range(0,len(feat)):
                    grad.increment_count((i)*len(indexer)+indexer.index_of(feat[i]), pred_err) 
    
                #SGD
                
                if True:
                    SGD.apply_gradient_update(grad,1)
                    weights = SGD.get_final_weights()
                
                #ADAgrad
                
                if False:
                    AdaGrad.apply_gradient_update(grad,1)
                    weights = AdaGrad.get_final_weights()
                if False:
                    Unreg_AdaGrad.apply_gradient_update(grad,1)
                    weights = Unreg_AdaGrad.get_final_weights()
        
    return PersonClassifier(weights,indexer)


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
