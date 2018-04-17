'''
Created on Apr 13, 2018

@author: adil
'''

'''
Neural Network for digit recognition from mnist dataset 
'''
import numpy as np


class Neural_Net(object):
    
    def __init__(self,sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = np.array([np.random.randn(y, 1) for y in sizes[1:]])
        self.weights =np.array( [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])])*0.01
    
    def forwardfeed(self,x):
        activation = x
        
        a_s = [x]             # list to store all the activations, layer by layer
        z_s = []              # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            z_s.append(z)
            activation = self.sigmoid(z)
            a_s.append(activation)
        a_s=np.array(a_s)
        z_s=np.array(z_s)
        
        return (z_s,a_s)
    
    def compute_cost(self,a,y):
    
        m = y.shape[1] # number of example

    
        logprobs = np.multiply(np.log(a),y) + np.multiply(np.log(1 - a),1 - y)
        cost = - np.sum(logprobs) * (1 / m)
    
    
        cost = np.squeeze(cost)     
    
    
        return cost




    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    
    

nn=Neural_Net([2,3,1])
 
x=np.array([[5,4,1],
            [8,9,3]])
   
nn.forwardfeed(x)