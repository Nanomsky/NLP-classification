#Helper functions for Logistic Regressions
#Ref: Cousera AI Deep Learning Course 2020

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys


sns.set()
###############################################################################
# Split Data
    
def splitdata(X, Y, rand_seed, tnx):
    '''
    Function used to split data into training, test and validation datastes
    This takes the predictor variables X and response variables Y, and 
    
    Input
    =====
        X         = An m by nx (nx = number of features) data matrix
        Y         = An m by 1 array of class labels
        rand_seed = Integer to ensure reproducibility for random generation 
        tnx       = Float between 0 and 1 used to specify the size of test/validation
    
    Output
    ======
        xtr, ytr = Training data, label
        xva, yva = Validation data, label
        xte, yte = Test data, label
    '''
    np.random.seed(rand_seed)
    m    = X.shape[0]
    index = np.random.permutation(m)
    
    if (tnx > 1) or (tnx < 0) :
        print("This should be greater than 0 and less than 1")

    len1= int(np.round(len(index)* tnx, 0))
    len2= int(np.round(len(index)* (1-tnx)/2, 0))

    xtr  = X[index[0:len1],:]
    xva  = X[index[len1:(len1 + len2)],:]
    xte  = X[index[(len1 + len2):],:]
    
    ytr  = Y[index[0:len1]]
    yva  = Y[index[len1:(len1 + len2)]]
    yte  = Y[index[(len1 + len2):]]
    
    print('{} training examples and {} features'.format(xtr.shape[0],xtr.shape[1]))
    print('{} validation examples and {} features'.format(xva.shape[0],xva.shape[1]))
    print('{} testing examples and {} features'.format(xte.shape[0],xte.shape[1]))
    
    return xtr, xva, xte, ytr.reshape(len(ytr),1), yva.reshape(len(yva),1), yte.reshape(len(yte),1)
####################################################
#plt.style.use('fivethirtyeight')
plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')

####################################################
#Initialise Parameters
def initalise_parameters(X):
    
    w = np.zeros(X.shape[0]).reshape(X.shape[0],1)
    b = np.zeros(1).reshape(1,1)

    return w,b

###################################################
#Compute Sigmoid Function
def sigmoid(x):
        
    z = 1/(1+ np.exp(-x))
    
    return z    

###################################################
#Compute Activation
def computeActivation(X, w, b):
    
    z  = np.dot(w.T,X) + b
    A  = sigmoid(z)
    
    return A

###################################################
#Propagate through X and compute gradient and Cost
def propagate(w, b, X, Y):
    
    m    = X.shape[1]
    A    = computeActivation(X, w, b)
    cost = - 1/m * np.sum((Y* np.log(A)) + ((1-Y) * np.log(1-A))) 
    dw   =   1/m * np.dot(X,(A - Y).T)
    db   =   1/m * np.sum(A-Y)
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost

###################################################
# Optimise parameters
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = True):
    
    costs = []
    for i in range(num_iterations):
        
        grads, cost = propagate(w, b, X, Y)
        dw = grads["dw"]
        db = grads["db"]
        w  = w - learning_rate * dw
        b  = b - learning_rate * db
        
        if i % 1000 == 0:
            costs.append(cost)
            
        if print_cost and i % 10000 == 0:
            print("Cost after iteration %i: %f" %(i, cost))
              
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

###################################################
#Make Prediction
def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T,X)  + b) 
    
    for i in range(A.shape[1]):
        if A[:,i] <= 0.5:
            Y_prediction[:,i] = 0
        elif A[:,i] > 0.5:
            Y_prediction[:,i] = 1
        pass
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction, A
#A is returned so we can estimate the ROC/AUC


####################################################
# Plot distribution of the datasets
def plot_dataset_distrib(xtr, xva, xte, ytr, yva, yte):
    
    num_of_train_pos = ytr[ytr==1].sum()
    num_of_val_pos   = yva[yva==1].sum()
    num_of_test_pos  = yte[yte==1].sum()
    num_of_train_neg = len(ytr) -  ytr[ytr==1].sum()
    num_of_val_neg   = len(yva) -  yva[yva==1].sum()
    num_of_test_neg  = len(yte) -  yte[yte==1].sum()
    
    print("Training set has {} positive and {} negative labels".format(num_of_train_pos ,num_of_train_neg))
    print("Validation set has {} positive and {} negative labels".format(num_of_val_pos,num_of_val_neg))
    print("Test set has {} positive and {} negative labels".format(num_of_test_pos,num_of_test_neg))
    print('\n')
   
    fig = plt.figure(figsize=(18,6))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    Train = [num_of_train_pos, num_of_train_neg]
    Val   = [num_of_val_pos,   num_of_val_neg] 
    Test  = [num_of_test_pos,  num_of_test_neg]

    Labels = ['pos', 'Neg']
    Explode = [0,0.1]

    ax1.pie(Train, explode=Explode, labels=Labels,shadow=True,startangle=45,autopct='%1.2f%%');
    ax1.set_title('Train set label distribution',fontsize=20);

    ax2.pie(Val, explode=Explode, labels=Labels ,shadow=True,startangle=45,autopct='%1.2f%%');
    ax2.set_title('Validation set label distribution',fontsize=20);

    ax3.pie(Test, explode=Explode, labels=Labels, shadow=True,startangle=45,autopct='%1.2f%%');
    ax3.set_title('Test set label distribution',fontsize=20);
    
    plt.show()
    print('\n')
    print('\n')
    #############
     #############
    fig2 = plt.figure(figsize=(10,5))
    split_data = ['Training', 'Validation','Test']
    pos_label  = [num_of_train_pos, num_of_val_pos, num_of_test_pos]
    neg_label  = [num_of_train_neg, num_of_val_neg,num_of_test_neg]

    index = np.arange(3)
    width = 0.30

    plt.bar(index,pos_label, width, color='maroon', label='Positive Label')
    plt.bar(index+width,neg_label, width, color='grey', label='Negative Label')
    plt.title("Labels",fontsize=20)

    #plt.xlabel("Data",fontsize=20)
    plt.ylabel("Number of values",fontsize=20)

    plt.xticks(index+width/2, split_data)

    plt.legend(loc='best')

    plt.show()

 #########################################################
#Transpose all data
def transposeAll(xtr, xva, xte, ytr, yva, yte):
    ytr=ytr.T
    yva=yva.T
    yte=yte.T
    xtr=xtr.T
    xva=xva.T
    xte=xte.T
        
    return xtr, xva, xte, ytr, yva, yte
    
    
