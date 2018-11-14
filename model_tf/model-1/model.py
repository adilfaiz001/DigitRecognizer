'''
Created on Oct 6, 2018

@author: adil
'''
from builtins import type

'''
Tensorflow implementation of CNN on MNIST dataset and studying model on different parameters.
2 layer CNN Model 

'''

import math 
import numpy as np
import matplotlib.pyplot as plt
import scipy
import tensorflow as tf
from tensorflow.python.framework import ops
from load_data import load_dataset
import pickle
import pandas as pd

np.random.seed(1)


#Loading the data
(train,validate,test) = load_dataset()

#training data
train_x,train_y = train
test_x,test_y = test
validate_x,validate_y = validate
#Example of a picture
index = 6
img = np.reshape(train_x[index],(28,28))
#plt.imshow(img)
#plt.show()

#placeholder function
def create_placeholders(n_H0,n_W0,n_C0,n_y):
    
    X = tf.placeholder(tf.float32,[None,n_H0,n_W0,n_C0], name="X")
    Y = tf.placeholder(tf.float32,[None,n_y])
    return X,Y

#Initialize Parametetrs
def initialize_parameters():
    
    W1 = tf.get_variable("W1",[4,4,1,8],initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2",[2,2,8,16],initializer=tf.contrib.layers.xavier_initializer(seed=0))
    
    parameters = {"W1":W1,
                  "W2":W2}
    
    return parameters

#Forward Propagation
def forward_propagation(X,parameters):
    '''
    (CONV2D -> RELU -> MAXPOOL) -> (CONV2D -> RELU -> MAXPOOL) -> FLATTEN -> FULLYCONNECTED
    
    '''
    
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
    
    
#Compute Cost 
def compute_cost(Z3,Y):
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=Z3))
    
    return cost 

#Random minibatches
def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples) (m, Hi, Wi, Ci)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples) (m, n_y)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[num_complete_minibatches * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches
        
        
#Model 
def model(X_train,Y_train,X_test,Y_test,X_validate,Y_validate,learning_rate=0.009,num_epochs=10,minibatch_size=64,print_cost=True):
    
    ops.reset_default_graph()  
    tf.set_random_seed(1)                             # to keep results consistent (tensorflow seed)
    seed = 3 
    
    (m, n_H0, n_W0, n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []
    
    X,Y = create_placeholders(n_H0, n_W0, n_C0, n_y)
    
    parameters = initialize_parameters()
    
    Z3 = forward_propagation(X, parameters)
    
    cost = compute_cost(Z3, Y)
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)    ##optimizer vallidate
    
    init = tf.global_variables_initializer()
    
    saver = tf.train.Saver()
    
    writer = tf.summary.FileWriter('./graphs/lecture02', tf.get_default_graph())
    with tf.Session() as sess:
        sess.run(init);
        writer.add_graph(sess.graph)
        
        for epoch in range(num_epochs):
            minibatch_cost = 0.
            
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            
            for minibatch in minibatches:
                (minibatch_x,minibatch_y) = minibatch 
                
                _, temp_cost = sess.run([optimizer,cost],feed_dict = {X:minibatch_x,Y:minibatch_y})   ##Main running line
                
                minibatch_cost += temp_cost / num_minibatches
                
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
    
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate) + "\nOptimizer:Adam")
        plt.show()

        # Calculate the correct predictions
        predict_op = tf.argmax(Z3, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        validate_accuracy = accuracy.eval({X: X_validate,Y: Y_validate})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
        print("Validate Accuracy:",validate_accuracy)
        
        save_path = saver.save(sess,"/W1/_stn_/workspace.py/deeplearning_projects_tf/src/mnist/model2/model/model_adam_tf.ckpt")
        print("Model saved in path:%s"%save_path)  
        
        df = pd.read_csv('./mnist/test.csv')
        X_data = df.values
        X_data = np.reshape(X_data,(X_data.shape[0],28,28,1))
        X_data = np.float32(X_data)
        y_pred = tf.argmax(sess.run(Z3,feed_dict={X:X_data}),1)
        
        '''
        print(y_pred.eval())
        
        lst_pred = np.empty((0,2),int)
        for col in range(1,y_pred.eval().shape[0]+1):
            
            lst_pred = np.append(lst_pred,[[col,y_pred.eval()[col-1]]], axis = 0)
            
        dataframe = pd.DataFrame(data = lst_pred,columns = ['ImageId','Label'])
        dataframe = dataframe.set_index('ImageId')
        dataframe.to_csv('Test_Labels.csv')
        '''
        return train_accuracy, test_accuracy, parameters
    
    
    
    
    

_, _, parameters = model(train_x, train_y, test_x, test_y,validate_x,validate_y)


    
    
    
    
    