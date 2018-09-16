'''
Created on Apr 13, 2018

@author: adil
'''

'''
Load MNIST dataset and extract into training,validation and test datasets.
Convert labels into vectorized form.
'''


import numpy as np 
import gzip
import pickle


def load_data():
    with gzip.open('mnist.pkl.gz','rb') as f:
        tr_d, va_d, te_d =pickle.load(f)
    
    train_x,train_y=tr_d
    train_y=np.array(np.squeeze([vectorized_result(y) for y in train_y],axis=0)) 
        
    training_data=(train_x,train_y)   
    
    
    val_x,val_y=va_d
    val_y = np.array(np.squeeze([vectorized_result(y) for y in val_y], axis=0))
    
    validation_data = (val_x,val_y)
    
    test_x,test_y = te_d
    test_y = np.array(np.squeeze([vectorized_result(y) for y in test_y],axis=0))
    
    test_data = (test_x,test_y)
    
    return (training_data,validation_data,test_data)
  
    
def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

