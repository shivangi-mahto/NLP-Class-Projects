# feedforward_example_pytorch.py

import torch
import torch.nn as nn
from torch import optim
import numpy as np
import random


# DEFINING THE COMPUTATION GRAPH
# Define the core neural network: one hidden layer, tanh nonlinearity
# Returns probabilities; in general your network can be set up to return probabilities, log probabilities,
# or (log) probabilities + loss
class FFNN(nn.Module):
    def __init__(self, inp, hid, out):
        super(FFNN, self).__init__()
        self.V = nn.Linear(inp, hid)
        self.g = nn.Tanh()
        self.W = nn.Linear(hid, out)
        self.softmax = nn.Softmax(dim=0)
        # Initialize weights according to the Xavier Glorot formula
        nn.init.xavier_uniform(self.V.weight)
        nn.init.xavier_uniform(self.W.weight)

    # Forward computation. Backward computation is done implicitly (nn.Module already has an implementation of
    # it that you shouldn't need to override)
    def forward(self, x):
        return self.softmax(self.W(self.g(self.V(x))))


# Form the input to the neural network. In general this may be a complex function that synthesizes multiple pieces
# of data, does some computation, handles batching, etc.
def form_input(x):
    return torch.from_numpy(x).float()


# Example of training a feedforward network with one hidden layer to solve XOR.
if __name__=="__main__":
    # MAKE THE DATA
    # Synthetic data for XOR: y = x0 XOR x1
    train_xs = np.array([[0, 0], [0, 1], [0, 1], [1, 0], [1, 0], [1, 1]], dtype=np.float32)
    train_ys = np.array([0, 1, 1, 1, 1, 0], dtype=np.float32)
    # Define some constants
    # Inputs are of size 2
    feat_vec_size = 2
    # Let's use 10 hidden units
    embedding_size = 10
    # We're using 2 classes. What's presented here is multi-class code that can scale to more classes, though
    # slightly more compact code for the binary case is possible.
    num_classes = 2

    # RUN TRAINING AND TEST
    num_epochs = 100
    ffnn = FFNN(feat_vec_size, embedding_size, num_classes)
    initial_learning_rate = 0.1
    optimizer = optim.Adam(ffnn.parameters(), lr=0.1)
    for epoch in range(0, num_epochs):
        ex_indices = [i for i in range(0, len(train_ys))]
        random.shuffle(ex_indices)
        total_loss = 0.0
        for idx in ex_indices:
            x = form_input(train_xs[idx])
            y = train_ys[idx]
            # Build one-hot representation of y
            y_onehot = torch.zeros(num_classes)
            #print (y_onehot.shape)
            y_onehot.scatter_(0, torch.from_numpy(np.asarray(y,dtype=np.long)), 1)
            #print (y_onehot.shape)
            # Zero out the gradients from the FFNN object. *THIS IS VERY IMPORTANT TO DO BEFORE CALLING BACKWARD()*
            ffnn.zero_grad()
            probs = ffnn.forward(x)
            # Can also use built-in NLLLoss as a shortcut here (takes log probabilities) but we're being explicit here
            print (probs,  y_onehot)
            loss = torch.neg(torch.log(probs)).dot(y_onehot)
            total_loss += loss
            loss.backward()
            optimizer.step()
        print("Loss on epoch %i: %f" % (epoch, total_loss))
    # Evaluate on the train set
    train_correct = 0
    for idx in range(0, len(train_xs)):
        # Note that we only feed in the x, not the y, since we're not training. We're also extracting different
        # quantities from the running of the computation graph, namely the probabilities, prediction, and z
        x = form_input(train_xs[idx])
        y = train_ys[idx]
        probs = ffnn.forward(x)
        prediction = torch.argmax(probs)
        if y == prediction:
            train_correct += 1
        print("Example " + repr(train_xs[idx]) + "; gold = " + repr(train_ys[idx]) + "; pred = " +\
              repr(prediction) + " with probs " + repr(probs))
    print(repr(train_correct) + "/" + repr(len(train_ys)) + " correct after training")
class FFNNnew(nn.Module):
    def __init__(self, inp, hid, out):
        super(FFNNnew, self).__init__()
        self.V = nn.Linear(inp, hid)
        self.g = nn.Tanh()
        self.W = nn.Linear(hid, out)
        self.softmax = nn.Softmax(dim=0)
        # Initialize weights according to the Xavier Glorot formula
        nn.init.xavier_uniform(self.V.weight)
        nn.init.xavier_uniform(self.W.weight)

    # Forward computation. Backward computation is done implicitly (nn.Module already has an implementation of
    # it that you shouldn't need to override)
    def forward(self, x):
        return self.softmax(self.W(self.g(self.V(x))))

    '''
                #print(train_data_batch[0])
            
            y_onehot.zero_()
            y_onehot.scatter_(1, torch.from_numpy(np.asarray(train_labels_batch,dtype=np.long)), 1)
            #print (train_labels_batch)
            #print (y_onehot)
            NET.zero_grad()
            output = NET.forward(train_data_batch)
            #loss function
            criterion = nn.BCELoss()
            loss = criterion(output, y_onehot)
            loss.backward()
            optimizer.step()
            
            print (output,y_onehot)   
            print(loss)
            total_loss += loss.data.numpy()
            
    #READ DEV DATA

    #print(seq_avg_emb.shape)
    sum2 = evaluate(NET, seq_avg_emb,tl2d)
    probs = NET.forward(form_input(seq_avg_emb_dev))
    #print(probs)
    sum_all = 0
    for i in range(0,len_dev_data):    
        prediction = torch.argmax(probs[i])
        #print(prediction.numpy()-dl2d[i])
        if(prediction.numpy()==dl2d[i]):
            sum_all = sum_all + 1
            #print(prediction.numpy(),dl2d[i])