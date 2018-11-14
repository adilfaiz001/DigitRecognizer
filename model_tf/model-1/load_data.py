'''
Created on Oct 6, 2018

@author: adil
'''
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

import numpy as np 
import gzip
import pickle
import tensorflow as tf

def load_dataset():
    with gzip.open('mnist.pkl.gz','rb') as f:
        tr_d, va_d, te_d =pickle.load(f,encoding='iso-8859-1')
        
    train_x, train_y = tr_d
    train_x = np.reshape(train_x,(train_x.shape[0],28,28,1))
    train_y = one_hot_matrix(train_y, 10)
    train_data = (train_x,train_y)
    
    validate_x,validate_y = va_d
    validate_x = np.reshape(validate_x,(validate_x.shape[0],28,28,1))
    validate_y = one_hot_matrix(validate_y,10)
    validation_data = (validate_x,validate_y)
    
    test_x,test_y = te_d
    test_x = np.reshape(test_x,(test_x.shape[0],28,28,1))
    test_y = one_hot_matrix(test_y,10)
    test_data = (test_x,test_y)
    
    return (train_data,validation_data,test_data)    
    


def one_hot_matrix(labels,C):
    
    c = tf.constant(C, name='classes')
    one_hot_matrix = tf.one_hot(indices=labels, depth=c,axis=0)
    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()
    
    return one_hot.T


load_dataset()