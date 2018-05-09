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
        self.b = np.array([np.random.randn(y, 1) for y in sizes[1:]])
        self.W =np.array( [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])])*0.01
        print self.W[0].shape
        print self.W[1].shape
        print self.b[0].shape
        print self.b[1].shape
    def forwardfeed(self,X):  
        
        activation = X
        activation_cache = [X]             # list to store all the activations, layer by layer
        linear_cache = []              # list to store all the z vectors, layer by layer
        
        
        for b, w in zip(self.b, self.W):
            z = np.dot(w, activation)+b
            linear_cache.append(z)
            activation = self.sigmoid(z)
            activation_cache.append(activation)
        
        
        activation_cache=np.array(activation_cache)
        linear_cache=np.array(linear_cache)
        
        caches=(linear_cache,activation_cache)
        AL=activation_cache[-1]
        
        return AL,caches
    
    def compute_cost(self,AL,Y):
    
        m = Y.shape[1] # number of example

        logprobs=Y*np.log(AL)+(1-Y)*(np.log(1-AL))
        cost = (-1.0 / m) * np.sum(logprobs)
        cost = np.squeeze(cost)     
    
        return cost
    

    def backwardpropagation(self,X,Y,caches):
        m=X.shape[1]
        
        db = [np.zeros(b.shape) for b in self.b]
        dw = [np.zeros(w.shape) for w in self.W]
        
        linear_cache,activation_cache=caches
        
        L=len(activation_cache)
        
        AL=activation_cache[-1]  
        
        dA_prev = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
       
        dz = AL-Y
        
        dw[-1] = (1.0/m)*np.dot(dz,activation_cache[-2].T) 
        
        db[-1] = (1.0/m)*np.sum(dz,axis=1,keepdims=True)
        
        dA_prev=np.dot(self.W[-1].T,dz)
        
        for i in reversed(xrange(L-2)):
            

            dz = dA_prev * self.sigmoid_prime(linear_cache[i])      #linear_cache storing z from index 0 for W1
            
            dw[i] = (1.0/m) * np.dot(dz,activation_cache[i].T)      # dw storing from 0 index for layer 1 i.e W1 ,take care for linear_cache and activation_cache as  index differ 
            
            db[i] = (1.0/m) * np.sum(dz,axis=1,keepdims=True)
            dA_prev = np.dot(self.W[i].T,dz)

        dw = np.array(dw)
        db = np.array(db)
        
        grads = (dw,db)
        
        return grads
    
    def update_parameters(self,grads,learning_rate):
        
        dw,db = grads
        L = len(dw)
        
        for i in xrange(L):
            self.W[i] = self.W[i] - learning_rate * dw[i]
            self.b[i] = self.b[i] - learning_rate * db[i]
        
            
            
            
        
        


    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    
    def sigmoid_prime(self,z):
        return self.sigmoid(z)*(1-self.sigmoid(z))
    
    

'''
nn=Neural_Net([2,3,1])
 
x=np.array([[5,4,1],
            [8,9,3]])
y=np.array([[1, 0, 1]])

for i in xrange(0,10000):
    AL,caches = nn.forwardfeed(x)
    cost=nn.compute_cost(AL, y)
    grads = nn.backwardpropagation(x, y, caches)
    
    print "Cost"+str(i+1),cost
    
    nn.update_parameters(grads, 0.07)
'''