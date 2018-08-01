'''
Created on Apr 16, 2018

@author: adil
'''


import numpy as np 
from load_data import load_data
from model1 import Neural_Net
import pickle

train_set,valid_set,test_set=load_data()

x,y=train_set

X=x.T
Y=y.T



NN=Neural_Net([784,30,10])

for i in xrange(2500):
    
    AL,caches = NN.forwardfeed(X)

    cost = NN.compute_cost(AL, Y)
    print "Cost"+str(i+1),cost
    print AL.shape
    grads=NN.backwardpropagation(X, Y, caches)

    NN.update_parameters(grads,0.07)


parameters=(NN.W,NN.b)

newfile = 'parameters.pk'
with open(newfile, 'wb') as fi:
    pickle.dump(parameters, fi)