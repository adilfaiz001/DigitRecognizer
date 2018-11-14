'''
Created on Oct 11, 2018

@author: adil
'''
import tensorflow as tf
from tensorflow.python.framework import ops
import pandas as pd
import csv
import numpy as np
from load_data import load_dataset

def create_placeholders(n_H0,n_W0,n_C0):
    X = tf.placeholder(tf.float32,[None,n_H0,n_W0,n_C0],name="X")
    
    return X

def parameters():
    W1 = tf.get_variable("W1",[4,4,1,8],initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2",[2,2,8,16],initializer=tf.contrib.layers.xavier_initializer(seed=0))
    
    parameters = {"W1":W1,
                  "W2":W2}
    
    return parameters

def forward_propagate(X,parameters):
    W1 = parameters['W1']
    W2 = parameters['W2']
    
    Z1 = tf.nn.conv2d(X,W1, strides=[1,1,1,1], padding="SAME")
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1, ksize=[1,8,8,1], strides=[1,8,8,1], padding="SAME")
    
    Z2 = tf.nn.conv2d(P1,W2,strides=[1,1,1,1],padding="SAME")
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2,ksize=[1,8,8,1],strides=[1,4,4,1],padding="SAME")
    
    P = tf.contrib.layers.flatten(P2)
    Z3 = tf.contrib.layers.fully_connected(P,10,activation_fn=None)
    
    return Z3


def application(X_data):
    
    ops.reset_default_graph()
    
    (m,n_H0,n_W0,n_C0) = X_data.shape
    
    X = create_placeholders(n_H0, n_W0, n_C0)
    
    W1 = tf.get_variable("W1",[4,4,1,8])
    W2 = tf.get_variable("W2",[2,2,8,16])
    
    parameters = {"W1":W1,"W2":W2}
    
    Z3 = forward_propagate(X, parameters)
    
    
    init = tf.global_variables_initializer()
    saver = tf.train.Saver({"W1":W1,"W2":W2})
    
    with tf.Session() as sess:
        sess.run(init)        
        saver.restore(sess, "/W1/_stn_/workspace.py/deeplearning_projects_tf/src/mnist/model2/model/model_adam_tf.ckpt")   
        
        Y = sess.run(Z3,feed_dict={X:X_data})
        predict_op = tf.argmax(Z3,1)
        print(predict_op)


(train,validate,test) = load_dataset()
test_x,test_y = test

y_pred = application(test_x)


'''
df = pd.read_csv('./mnist/test.csv')
X = df.values
X = np.reshape(X,(X.shape[0],28,28,1))
X = np.float32(X)
y_pred = application(X)
lst_pred = np.empty((0,2),int)
for col in range(1,y_pred.shape[0]+1):
    
    lst_pred = np.append(lst_pred,[[col,y_pred[col-1]]], axis = 0)
    
dataframe = pd.DataFrame(data = lst_pred,columns = ['ImageId','Label'])
dataframe = dataframe.set_index('ImageId')
dataframe.to_csv('Test_Labels.csv')
'''