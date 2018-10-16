'''
Created on May 10, 2018

@author: adil
'''

'''
ANN for mnist dataset with batch gradient and different optimization algorithms 
ADAM
RMSProp
Momentum
'''


import numpy as np
import math 
import sklearn 
import matplotlib.pyplot as plt
import pickle
from load_data import load_dataset

np.seterr(divide='ignore', invalid='ignore')

class MiniBatch_ANN(object):
    
    def __init__(self,sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.b = np.array([np.random.randn(y, 1) for y in sizes[1:]])
        self.W =np.array( [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])])*0.01
        
    
    #=============================Randomized mini batches=============================#
    def random_mini_batches(self,X,Y,mini_batch_size=64,seed = 0):
        
        np.random.seed(seed)
        m = X.shape[1]
        mini_batches = []
        
        
        permutation = list(np.random.permutation(m))
        shuffled_X = X[:, permutation]
        shuffled_Y = Y[:, permutation].reshape((10,m))
        
        num_complete_batches = int(m/mini_batch_size)
        
        for k in range(0,num_complete_batches):
            mini_batch_X = shuffled_X[:, k * mini_batch_size:(k+1) * mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k * mini_batch_size:(k+1) * mini_batch_size]
            
            mini_batch = (mini_batch_X,mini_batch_Y)
            mini_batches.append(mini_batch)
        
        if m % mini_batch_size != 0 :
            end = m - mini_batch_size * math.floor(m / mini_batch_size)
            mini_batch_X = shuffled_X[:, num_complete_batches * mini_batch_size:]
            mini_batch_Y = shuffled_Y[:, num_complete_batches * mini_batch_size:]
            
            mini_batch = (mini_batch_X,mini_batch_Y)
            mini_batches.append(mini_batch)
        
        return mini_batches
    #=====================================================================================#
    
    
    
    #===============================Neural Network========================================#
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
        
        try:
            dA_prev = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        except:
            print "Exception Occured"
            return 
        
       
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
    #=====================================================================================#
    
    
    #============================Update Parameters========================================#
     
    def update_parameters(self,grads,learning_rate):
        
        dw,db = grads
        L = len(dw)
        
        for i in xrange(L-1):
            self.W[i] = self.W[i] - learning_rate * dw[i]
            self.b[i] = self.b[i] - learning_rate * db[i]      

    #-----------------------------Momentum------------------------------------------------#
    def initialize_velocity(self):
        L = self.num_layers
        v = {}
        for l in xrange(L-1):
            v["dW" + str(l+1)] = np.zeros_like(self.W[l])
            v["db" + str(l+1)] = np.zeros_like(self.b[l])
        
        return v
    
    def update_parameters_momentum(self,grads,v,beta,learning_rate):
        
        L = self.num_layers
        dW,db = grads
        
        for l in xrange(L-1):
            
            v["dW" + str(l+1)] = beta * v["dW" + str(l+1)] + (1-beta) * dW[l]         ## Check for self.W dimension with velocity dimensions
            v["db" + str(l+1)] = beta * v["db" + str(l+1)] + (1-beta) * db[l]
            
            self.W[l] = self.W[l] - learning_rate * v["dW" + str(l+1)]
            self.b[l] = self.b[l] - learning_rate * v["db" + str(l+1)]
        
        parameters = (self.W,self.b)
        
        return v
    #-------------------------------------------------------------------------------------#
    
    
    #---------------------------------Adam Optimization-----------------------------------#
    def initialize_adam(self):
        
        L = self.num_layers
        v = {}
        s = {}
        
        for l in xrange(L-1):
            
            v["dW" + str(l + 1)] = np.zeros_like(self.W[l])
            v["db" + str(l + 1)] = np.zeros_like(self.b[l])
            
            s["dW" + str(l + 1)] = np.zeros_like(self.W[l])
            s["db" + str(l + 1)] = np.zeros_like(self.b[l])
            
        return v,s
    
    def update_parameters_adam(self,grads,v,s,t,learning_rate=0.01,beta1=0.9,beta2=0.999,epsilon=1e-8):
        
        L = self.num_layers
        v_corrected = {}
        s_corrected = {}
        
        dW,db = grads
        
        for l in xrange(L-1):
            
            v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1 - beta1) * dW[l]
            v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1 - beta1) * db[l]
            
            v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - np.power(beta1, t))
            v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - np.power(beta1, t))
            
            s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1 - beta2) * np.power(dW[l],2)
            s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1 - beta2) * np.power(db[l],2)
            
            s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - np.power(beta2,t))
            s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - np.power(beta2,t))
            
            self.W[l] = self.W[l] - learning_rate * v_corrected["dW" + str(l + 1)] / np.sqrt(s_corrected["dW" + str(l + 1)] + epsilon)
            self.b[l] = self.b[l] - learning_rate * v_corrected["db" + str(l + 1)] / np.sqrt(s_corrected["db" + str(l + 1)] + epsilon)
            
        
        parameters = (self.W,self.b)
        
        return v, s
    #------------------------------------------------------------------------------------#
    
    #====================================================================================#
    
    
    
    #=============================computation function===============================##
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    
    def sigmoid_prime(self,z):
        return self.sigmoid(z)*(1-self.sigmoid(z))
    
    def relu(self,z):
        return z * (z > 0)

    def relu_prime(self,z):
        return 1. * (z > 0)
    
    #=================================================================================##
    
    
    
    def model(self, X, Y, layers_dims, optimizer, learning_rate = 0.0007, mini_batch_size=64, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=10000, print_cost=True):
        
        """
        L-layer neural network model which can be run in different optimizer modes.
        
        Arguments:
        X -- input data, of shape (2, number of examples)
        Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
        layers_dims -- python list, containing the size of each layer
        learning_rate -- the learning rate, scalar.
        mini_batch_size -- the size of a mini batch
        beta -- Momentum hyperparameter
        beta1 -- Exponential decay hyperparameter for the past gradients estimates 
        beta2 -- Exponential decay hyperparameter for the past squared gradients estimates 
        epsilon -- hyperparameter preventing division by zero in Adam updates
        num_epochs -- number of epochs
        print_cost -- True to print the cost every 1000 epochs
    
        Returns:
        parameters -- python dictionary containing your updated parameters 
        """
        L = len(layers_dims)
        costs = []
        t = 0 
        seed = 1
        
        #Initialize optimizer
        if optimizer == "gd":
            pass
        elif optimizer == "momentum":
            v = self.initialize_velocity()
        
        elif optimizer == "adam":
            v,s = self.initialize_adam()
                 
        #optimization loop
        for i in xrange(num_epochs):
            
            seed = seed + 1
            mini_batches = self.random_mini_batches(X, Y, mini_batch_size, seed)
            
            for mini_batch in mini_batches:
                
                (mini_batch_X,mini_batch_Y) = mini_batch
                
                AL, caches = self.forwardfeed(mini_batch_X)
                
                cost = self.compute_cost(AL, mini_batch_Y)
                
                grads = self.backwardpropagation(mini_batch_X, mini_batch_Y, caches)
                
                #Update Parameters 
                if optimizer == "gd":
                    self.update_parameters(grads, learning_rate)
                elif optimizer == "momentum":
                    v = self.update_parameters_momentum(grads,v,beta,learning_rate)
                elif optimizer == "adam":
                    t = t + 1
                    v,s = self.update_parameters_adam(grads,v,s,t,learning_rate,beta1,beta2,epsilon)
                    
            if print_cost and i % 1000 == 0:
                print "Cost after epoch %i : %f" %(i, cost)
            if print_cost and i % 100 == 0:
                costs.append(cost) 
            
            print i      # to see ending not freezing
        
        
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('epochs(per 100)')
        plt.title('Learning rate = '+str(learning_rate))
        plt.show()  
        
        parameters = (self.W,self.b)
        
        newfile = 'parameters_adam.pk'
        with open(newfile, 'wb') as fi:
            pickle.dump(parameters, fi)
        





train_set,valid_set,test_set=load_data()

X,Y = train_set
X,Y = X.T,Y.T

layers_dims = [784,84,10]
nn = MiniBatch_ANN(layers_dims)
nn.model(X, Y, layers_dims,"adam")






