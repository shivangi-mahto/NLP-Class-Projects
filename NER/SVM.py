# models.py

from nerdata import *
from utils import *
from optimizers import *
import numpy as np


# Scoring function for sequence models based on conditional probabilities.
# Scores are provided for three potentials in the model: initial scores (applied to the first tag),
# emissions, and transitions. Note that CRFs typically don't use potentials of the first type.
class ProbabilisticSequenceScorer(object):
    def __init__(self, tag_indexer, word_indexer, init_log_probs, transition_log_probs, emission_log_probs):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    def score_init(self, sentence, tag_idx):
        return self.init_log_probs[tag_idx]

    def score_transition(self, sentence, prev_tag_idx, curr_tag_idx):
        return self.transition_log_probs[prev_tag_idx, curr_tag_idx]

    def score_emission(self, sentence, tag_idx, word_posn):
        word = sentence.tokens[word_posn].word
        word_idx = self.word_indexer.index_of(word) if self.word_indexer.contains(word) else self.word_indexer.get_index("UNK")
        return self.emission_log_probs[tag_idx, word_idx]


class FeatureBasedSequenceScorer(object): # REMEMBER TO RETURN LOG OF SCORES INSEAD OF SCORES ITSELF
    def __init__(self, tag_indexer, word_indexer,weights):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.weights = weights

    def score_transition(self, transit_feat):
        score_transit = 0 
        for  i in range(0,len(transit_feat)):
            score_transit = score_transit + self.weights[transit_feat[i]]
        log_phi_transit = (score_transit)
        return log_phi_transit

    def score_emission(self, emission_feat):
        score_emiss = 0.0 
        for  i in range(0,len(emission_feat)):
            score_emiss = score_emiss + self.weights[emission_feat[i]]
        log_phi_emission = (score_emiss)
        return log_phi_emission


class HmmNerModel(object):
    def __init__(self, tag_indexer, word_indexer, init_log_probs, transition_log_probs, emission_log_probs):
        self.tag_indexer = tag_indexer
        self.word_indexer = word_indexer
        self.init_log_probs = init_log_probs
        self.transition_log_probs = transition_log_probs
        self.emission_log_probs = emission_log_probs

    # Takes a LabeledSentence object and returns a new copy of that sentence with a set of chunks predicted by
    # the HMM model. See BadNerModel for an example implementation
    def decode(self, sentence):
        
        Probability = ProbabilisticSequenceScorer(self.tag_indexer,self.word_indexer, self.init_log_probs,self.transition_log_probs, self.emission_log_probs )
       
        score = np.zeros([len(sentence),len(self.tag_indexer)]) # keep score for all s in all stages of i
        prev_tag_max = np.empty((len(sentence) , len(self.tag_indexer))) # pointer for argmax previous tag
        
        for s in range(0,len(self.tag_indexer)):
            word_0 = sentence.tokens[0].word
            #score[0][s] = self.init_log_probs[s] + self.emission_log_probs[s][self.word_indexer.index_of(word_0)] 
            score[0][s] = Probability.score_init(sentence, s) + Probability.score_emission(sentence,s,0)
        
        for i in range(1,len(sentence)):
            word_i = sentence.tokens[i].word
            for s in range(0,len(self.tag_indexer)):
                temp_score = 1*np.zeros(len(self.tag_indexer))
                for prev_tag in range(0,len(self.tag_indexer)):
                  temp_score[prev_tag] = score[i-1][prev_tag] + Probability.score_transition(sentence,prev_tag,s) + Probability.score_emission(sentence,s,i)
                score[i][s]= np.max(temp_score)
                prev_tag_max[i][s] = np.argmax(temp_score) 

        pred_tag = []
        pred_tag_ind = np.empty(len(sentence),dtype=int)

        pred_tag_ind[len(sentence)-1] = np.argmax(score[len(sentence)-1])
        for i in range(1,len(sentence)):
            pred_tag_ind[len(sentence)-i-1] = prev_tag_max[len(sentence)-i][pred_tag_ind[len(sentence)-i]]
        
        for i in range(0,len(sentence)):
            pred_tag.append(self.tag_indexer.get_object(pred_tag_ind[i]))

        return LabeledSentence(sentence.tokens, chunks_from_bio_tag_seq(pred_tag))


# Uses maximum-likelihood estimation to read an HMM off of a corpus of sentences.
# Any word that only appears once in the corpus is replaced with UNK. A small amount
# of additive smoothing is applied to
def train_hmm_model(sentences):
    # Index words and tags. We do this in advance so we know how big our
    # matrices need to be.
    tag_indexer = Indexer()
    word_indexer = Indexer()
    word_indexer.get_index("UNK")
    word_counter = Counter()
    for sentence in sentences:
        for token in sentence.tokens:
            word_counter.increment_count(token.word, 1.0)
    for sentence in sentences:
        for token in sentence.tokens:
            # If the word occurs fewer than two times, don't index it -- we'll treat it as UNK
            get_word_index(word_indexer, word_counter, token.word)
        for tag in sentence.get_bio_tags():
            tag_indexer.get_index(tag)
    # Count occurrences of initial tags, transitions, and emissions
    # Apply additive smoothing to avoid log(0) / infinities / etc.
    init_counts = np.ones((len(tag_indexer)), dtype=float) * 0.001
    transition_counts = np.ones((len(tag_indexer), len(tag_indexer)), dtype=float) * 0.001
    emission_counts = np.ones((len(tag_indexer), len(word_indexer)), dtype=float) * 0.001
    for sentence in sentences:
        bio_tags = sentence.get_bio_tags()
        for i in range(0, len(sentence)):
            tag_idx = tag_indexer.get_index(bio_tags[i])
            word_idx = get_word_index(word_indexer, word_counter, sentence.tokens[i].word)
            emission_counts[tag_idx][word_idx] += 1.0
            if i == 0:
                init_counts[tag_indexer.get_index(bio_tags[i])] += 1.0
            else:
                transition_counts[tag_indexer.get_index(bio_tags[i - 1])][tag_idx] += 1.0
    # Turn counts into probabilities for initial tags, transitions, and emissions. All
    # probabilities are stored as log probabilities
    #print(repr(init_counts))
    init_counts = np.log(init_counts / init_counts.sum())
    # transitions are stored as count[prev state][next state], so we sum over the second axis
    # and normalize by that to get the right conditional probabilities
    transition_counts = np.log(transition_counts / transition_counts.sum(axis=1)[:, np.newaxis])
    # similar to transitions
    emission_counts = np.log(emission_counts / emission_counts.sum(axis=1)[:, np.newaxis])
    
#    print("Tag indexer: %s" % tag_indexer)
#    print("Initial state log probabilities: %s" % init_counts)
#    print("Transition log probabilities: %s" % transition_counts)
#    print("Emission log probs too big to print...")
#    print("Emission log probs for India: %s" % emission_counts[:, word_indexer.get_index("India")])
#    print("Emission log probs for Phil: %s" % emission_counts[:, word_indexer.get_index("Phil")])
#    print("   note that these distributions don't normalize because it's p(word|tag) that normalizes, not p(tag|word)")
    return HmmNerModel(tag_indexer, word_indexer, init_counts, transition_counts, emission_counts)


# Retrieves a word's index based on its count. If the word occurs only once, treat it as an "UNK" token
# At test time, unknown words will be replaced by UNKs.
def get_word_index(word_indexer, word_counter, word):
    if word_counter.get_count(word) < 1.5:
        return word_indexer.get_index("UNK")
    else:
        return word_indexer.get_index(word)


class CrfNerModel(object):
    def __init__(self, tag_indexer, feature_indexer, feature_weights, transition_counts):
        self.tag_indexer = tag_indexer
        self.feature_indexer = feature_indexer
        self.feature_weights = feature_weights
        self.transition_counts = transition_counts

    # Takes a LabeledSentence object and returns a new copy of that sentence with a set of chunks predicted by
    # the CRF model. See BadNerModel for an example implementation
    def decode(self, sentence):
        prev_tag_max = np.empty((len(sentence) , len(self.tag_indexer))) # pointer for argmax previous tag      
        Scorer = FeatureBasedSequenceScorer(self.tag_indexer, self.feature_indexer, self.feature_weights)
        score = np.zeros([len(sentence),len(self.tag_indexer)]) 
        
        for s in range(0,len(self.tag_indexer)):
            emission_feat = extract_emission_features(sentence,0, self.tag_indexer.get_object(s), self.feature_indexer, add_to_indexer=False)
            score[0][s] = Scorer.score_emission(emission_feat)
        #2. for y_i with viterbi kind of - also extending to forward backward
        for i in range(1,len(sentence)):
            for s in range(0,len(self.tag_indexer)):
                emission_feat = extract_emission_features(sentence, i, self.tag_indexer.get_object(s), self.feature_indexer, add_to_indexer=False)
                temp_score = 1*np.zeros(len(self.tag_indexer))
                for prev_tag in range(0,len(self.tag_indexer)):
                  temp_score[prev_tag] = score[i-1][prev_tag] + self.transition_counts[prev_tag,s] + Scorer.score_emission(emission_feat)
                  
                score[i][s]= np.max(temp_score)
                prev_tag_max[i][s] = np.argmax(temp_score) 

        pred_tag = []
        pred_tag_ind = np.empty(len(sentence),dtype=int)

        pred_tag_ind[len(sentence)-1] = np.argmax(score[len(sentence)-1])
        for i in range(1,len(sentence)):
            pred_tag_ind[len(sentence)-i-1] = prev_tag_max[len(sentence)-i][pred_tag_ind[len(sentence)-i]]
        
        for i in range(0,len(sentence)):
            pred_tag.append(self.tag_indexer.get_object(pred_tag_ind[i]))
    
        return LabeledSentence(sentence.tokens, chunks_from_bio_tag_seq(pred_tag))


# Trains a CrfNerModel on the given corpus of sentences.
def train_crf_model(sentences):

    tag_indexer = Indexer()
    for sentence in sentences:
        for tag in sentence.get_bio_tags():
            tag_indexer.get_index(tag)
    print("Extracting features")
    feature_indexer = Indexer()
    # 4-d list indexed by sentence index, word index, tag index, feature index
    feature_cache = [[[[] for k in range(0, len(tag_indexer))] for j in range(0, len(sentences[i]))] for i in range(0, len(sentences))]
    
    hmm = train_hmm_model(sentences)
    Probability = ProbabilisticSequenceScorer(hmm.tag_indexer,hmm.word_indexer, hmm.init_log_probs,hmm.transition_log_probs, hmm.emission_log_probs )
    
    for sentence_idx in range(0,len(sentences)): #len(sentences)
        if sentence_idx % 10000 == 0:
            print("Ex %i/%i" % (sentence_idx, len(sentences)))
        for word_idx in range(0, len(sentences[sentence_idx])):
            for tag_idx in range(0, len(tag_indexer)):
                feature_cache[sentence_idx][word_idx][tag_idx] = extract_emission_features(sentences[sentence_idx], word_idx, tag_indexer.get_object(tag_idx), feature_indexer, add_to_indexer=True)
                 
    lr = 0.1
    #weights = 0.1*np.random.rand(len(feature_indexer))
    weights = 1*np.zeros(len(feature_indexer))
    SGD = SGDOptimizer(weights, lr)
    Unreg_AdaGrad = UnregularizedAdagradTrainer(weights)
    # forward backward implementation
    for epoch in range(0,5):
        for sentence_idx in range(0,len(sentences)): #len(sentences)
            sentence = sentences[sentence_idx] 
            #print (sentence)
            Scorer = FeatureBasedSequenceScorer(tag_indexer,feature_indexer,weights)

            alpha       = np.zeros([len(sentence),len(tag_indexer)])
            beta        = np.zeros([len(sentence),len(tag_indexer)])
            Marg_prob   = np.zeros([len(sentence),len(tag_indexer)])
            
            # for i = 0 
            for s in range(0,len(tag_indexer)):
                emission_feat = feature_cache[sentence_idx][0][s]
                alpha[0][s] = Scorer.score_emission(emission_feat) #+ Probability.score_init(sentence, s) #wTfe
                beta[len(sentence)-1][s] = np.log(1.0) #
            
            
            #2. for y_i with viterbi kind of - extending to forward backward
            for i in range(1,len(sentence)):            
                for s in range(0,len(tag_indexer)):
                    emission_feat = feature_cache[sentence_idx][i][s]
                    temp_alpha = np.zeros(len(tag_indexer))
                    
                    prev_tag = 0
                    temp_alpha[prev_tag] = alpha[i-1][prev_tag] + Scorer.score_emission(emission_feat) + Probability.score_transition(sentence,prev_tag,s) #Scorer.score_transition(prev_tag,s) + 
                    alpha[i][s] = temp_alpha[prev_tag]
                    for prev_tag in range(1,len(tag_indexer)):
                        temp_alpha[prev_tag] = alpha[i-1][prev_tag] + Scorer.score_emission(emission_feat) +  Probability.score_transition(sentence,prev_tag,s) #Scorer.score_transition(prev_tag,s) + 
                        alpha[i][s]= np.logaddexp(alpha[i][s],temp_alpha[prev_tag])

            for i in range(1,len(sentence)):
                j = len(sentence)-1-i
                for s in range(0,len(tag_indexer)):
                    temp_beta = np.zeros(len(tag_indexer))
                    #print("i,j",i,j)
                    next_tag = 0
                    temp_beta[next_tag] = beta[j+1][next_tag] + Scorer.score_emission(feature_cache[sentence_idx][j+1][next_tag]) + Probability.score_transition(sentence,s, next_tag) #+ Scorer.score_transition(s,next_tag)
                    #print(beta[j+1][next_tag])
                    beta[j][s] = temp_beta[next_tag]
                    for next_tag in range(1,len(tag_indexer)):
                        temp_beta[next_tag] = beta[j+1][next_tag] + Scorer.score_emission(feature_cache[sentence_idx][j+1][next_tag]) + Probability.score_transition(sentence,s, next_tag) #+ Scorer.score_transition(s,next_tag)
                        #print(beta[j+1][next_tag])
                        beta[j][s]= np.logaddexp(beta[j][s], temp_beta[next_tag])
       
            bio_tags = sentence.get_bio_tags()
            grad=Counter()
            for i in range(0,len(sentence)):
                # compute marginal probs using forward backward
                for s in range(0,len(tag_indexer)):
                    Marg_prob[i][s] = np.exp((alpha[i][s] + beta[i][s]) - np.log(np.sum(np.exp(alpha[i]+beta[i]))) )
                
                tag_idx = tag_indexer.get_index(bio_tags[i])
                feat_gold = feature_cache[sentence_idx][i][tag_idx]
                
                for j in range(0,len(feat_gold)):
                    grad.increment_count(feat_gold[j], 1) 
                    # grad with emission features only
                    for s in range(0,len(tag_indexer)):
                        feat_s = feature_cache[sentence_idx][i][s]
                        grad.increment_count(feat_s[j], -1*Marg_prob[i][s])
            #SGD
            #SGD.apply_gradient_update(grad,1)
            #weights = SGD.get_final_weights()
            
            #ADAGrad
            Unreg_AdaGrad.apply_gradient_update(grad,1)#len(sentence)
            weights = Unreg_AdaGrad.get_final_weights()
        print("weights in i%d",weights, epoch)
                    #prev_tag_max[i][s] = np.argmax(temp_score) 

    return CrfNerModel(tag_indexer, feature_indexer, weights, hmm.transition_log_probs)


# Extracts emission features for tagging the word at word_index with tag.
# add_to_indexer is a boolean variable indicating whether we should be expanding the indexer or not:
# this should be True at train time (since we want to learn weights for all features) and False at
# test time (to avoid creating any features we don't have weights for).
def extract_emission_features(sentence, word_index, tag, feature_indexer, add_to_indexer):
    feats = []
    curr_word = sentence.tokens[word_index].word
    # Lexical and POS features on this word, the previous, and the next (Word-1, Word0, Word1)
    for idx_offset in range(-1, 2):
        if word_index + idx_offset < 0:
            active_word = "<s>"
        elif word_index + idx_offset >= len(sentence):
            active_word = "</s>"
        else:
            active_word = sentence.tokens[word_index + idx_offset].word
        if word_index + idx_offset < 0:
            active_pos = "<S>"
        elif word_index + idx_offset >= len(sentence):
            active_pos = "</S>"
        else:
            active_pos = sentence.tokens[word_index + idx_offset].pos
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Word" + repr(idx_offset) + "=" + active_word)
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":Pos" + repr(idx_offset) + "=" + active_pos)
    # Character n-grams of the current word
    max_ngram_size = 3
    for ngram_size in range(1, max_ngram_size+1):
        start_ngram = curr_word[0:min(ngram_size, len(curr_word))]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":StartNgram=" + start_ngram)
        end_ngram = curr_word[max(0, len(curr_word) - ngram_size):]
        maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":EndNgram=" + end_ngram)
    # Look at a few word shape features
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":IsCap=" + repr(curr_word[0].isupper()))
    # Compute word shape
    new_word = []
    for i in range(0, len(curr_word)):
        if curr_word[i].isupper():
            new_word += "X"
        elif curr_word[i].islower():
            new_word += "x"
        elif curr_word[i].isdigit():
            new_word += "0"
        else:
            new_word += "?"
    maybe_add_feature(feats, feature_indexer, add_to_indexer, tag + ":WordShape=" + repr(new_word))
    return np.asarray(feats, dtype=int)

#Expected reuslts for HMM
    #Labeled F1: 76.89, precision: 4154/4862 = 85.44, recall: 4154/5943 = 69.90
#Expextec results for CRF with training epoch 5
    #Labeled F1: 88.34, precision: 5202/5834 = 89.17, recall: 5202/5943 = 87.53
class SVMModel(object):
    def __init__(self, tag_indexer, feature_indexer, feature_weights):
        self.tag_indexer = tag_indexer
        self.feature_indexer = feature_indexer
        self.feature_weights = feature_weights
        #self.transition_counts = transition_counts

    # Takes a LabeledSentence object and returns a new copy of that sentence with a set of chunks predicted by
    # the CRF model. See BadNerModel for an example implementation
    def decode(self, sentence):
        prev_tag_max = np.empty((len(sentence) , len(self.tag_indexer))) # pointer for argmax previous tag      
        Scorer = FeatureBasedSequenceScorer(self.tag_indexer, self.feature_indexer, self.feature_weights)
        score = np.zeros([len(sentence),len(self.tag_indexer)]) 
        
        for s in range(0,len(self.tag_indexer)):
            emission_feat = extract_emission_features(sentence,0, self.tag_indexer.get_object(s), self.feature_indexer, add_to_indexer=False)
            score[0][s] = Scorer.score_emission(emission_feat)
        #2. for y_i with viterbi kind of - also extending to forward backward
        for i in range(1,len(sentence)):
            for s in range(0,len(self.tag_indexer)):
                emission_feat = extract_emission_features(sentence, i, self.tag_indexer.get_object(s), self.feature_indexer, add_to_indexer=False)
                temp_score = 1*np.zeros(len(self.tag_indexer))
                for prev_tag in range(0,len(self.tag_indexer)):
                  temp_score[prev_tag] = score[i-1][prev_tag] + Scorer.score_emission(emission_feat)
                  
                score[i][s]= np.max(temp_score)
                prev_tag_max[i][s] = np.argmax(temp_score) 

        pred_tag = []
        pred_tag_ind = np.empty(len(sentence),dtype=int)

        pred_tag_ind[len(sentence)-1] = np.argmax(score[len(sentence)-1])
        for i in range(1,len(sentence)):
            pred_tag_ind[len(sentence)-i-1] = prev_tag_max[len(sentence)-i][pred_tag_ind[len(sentence)-i]]
        
        for i in range(0,len(sentence)):
            pred_tag.append(self.tag_indexer.get_object(pred_tag_ind[i]))
    
        return LabeledSentence(sentence.tokens, chunks_from_bio_tag_seq(pred_tag))

def train_svm_model(sentences):

    tag_indexer = Indexer()
    for sentence in sentences:
        for tag in sentence.get_bio_tags():
            tag_indexer.get_index(tag)
    print("Extracting features")
    feature_indexer = Indexer()
    # 4-d list indexed by sentence index, word index, tag index, feature index
    feature_cache = [[[[] for k in range(0, len(tag_indexer))] for j in range(0, len(sentences[i]))] for i in range(0, len(sentences))]
    
    #hmm = train_hmm_model(sentences)
    #Probability = ProbabilisticSequenceScorer(hmm.tag_indexer,hmm.word_indexer, hmm.init_log_probs,hmm.transition_log_probs, hmm.emission_log_probs )
    
    for sentence_idx in range(0,len(sentences)): #len(sentences)
        if sentence_idx % 10000 == 0:
            print("Ex %i/%i" % (sentence_idx, len(sentences)))
        for word_idx in range(0, len(sentences[sentence_idx])):
            for tag_idx in range(0, len(tag_indexer)):
                feature_cache[sentence_idx][word_idx][tag_idx] = extract_emission_features(sentences[sentence_idx], word_idx, tag_indexer.get_object(tag_idx), feature_indexer, add_to_indexer=True)
                 
    lr = 0.1
    #weights = 0.1*np.random.rand(len(feature_indexer))
    weights = 1*np.zeros(len(feature_indexer))
    SGD = SGDOptimizer(weights, lr)
    Unreg_AdaGrad = UnregularizedAdagradTrainer(weights)
    # forward backward implementation
    for epoch in range(0,5):
        for sentence_idx in range(0,len(sentences)): #len(sentences)
            sentence = sentences[sentence_idx] 
            #print (sentence)
            Scorer = FeatureBasedSequenceScorer(tag_indexer,feature_indexer,weights)

            prev_tag_max = np.empty((len(sentence),  len(tag_indexer))) # pointer for argmax previous tag      
            alpha        = np.zeros([len(sentence),  len(tag_indexer)])
            
            
            # for i = 0 
            for s in range(0,len(tag_indexer)):
                emission_feat = feature_cache[sentence_idx][0][s]
                alpha[0][s] = Scorer.score_emission(emission_feat) + 1 - int(np.equal(s,tag_idx)) #wTfe            
            
            #2. for y_i with viterbi kind of - extending to forward backward
            for i in range(1,len(sentence)):     
                bio_tags = sentence.get_bio_tags()
                tag_idx = bio_tags[i]
                for s in range(0,len(tag_indexer)):
                    
                    emission_feat = feature_cache[sentence_idx][i][s]
                    temp_alpha = np.zeros(len(tag_indexer))
                    
                 
                    for prev_tag in range(0,len(tag_indexer)):
                        temp_alpha[prev_tag] = alpha[i-1][prev_tag] + Scorer.score_emission(emission_feat) + 1 - int(np.equal(s,tag_idx))  #Scorer.score_transition(prev_tag,s) + 
                        #alpha[i][s]= np.logaddexp(alpha[i][s],temp_alpha[prev_tag])
                    alpha[i][s]= np.max(temp_alpha)
                    prev_tag_max[i][s] = np.argmax(temp_alpha) 

            pred_tag_ind = np.empty(len(sentence),dtype=int)

            pred_tag_ind[len(sentence)-1] = np.argmax(score[len(sentence)-1])
            for i in range(1,len(sentence)):
                pred_tag_ind[len(sentence)-i-1] = prev_tag_max[len(sentence)-i][pred_tag_ind[len(sentence)-i]]

            
            grad=Counter()
            for i in range(0,len(sentence)):
               
                tag_idx = tag_indexer.get_index(bio_tags[i])
                feat_gold = feature_cache[sentence_idx][i][tag_idx]
                
                for j in range(0,len(feat_gold)):
                    grad.increment_count(feat_gold[j], -1) 

                    feat_s = feature_cache[sentence_idx][i][pred_tag_ind[i]]
                    grad.increment_count(feat_s[j], 1)
            #SGD
            #SGD.apply_gradient_update(grad,1)
            #weights = SGD.get_final_weights()
            
            #ADAGrad
            Unreg_AdaGrad.apply_gradient_update(grad,1)#len(sentence)
            weights = Unreg_AdaGrad.get_final_weights()
        print("weights in i%d",weights, epoch)
                    #prev_tag_max[i][s] = np.argmax(temp_score) 

    return SVMModel(tag_indexer, feature_indexer, weights)
