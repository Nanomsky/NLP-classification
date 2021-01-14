#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 02:29:41 2021

@author: osita
"""
#Helper functions for NLP
#Ref: Cousera AI NLP course 2020

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys


sns.set()
###############################################################################
# Build frequency vocab
def build_freq(examples, label):
    label_list = np.squeeze(label).tolist() #Converts the label to list
    freq={}
    
    for lab, Text in zip(label_list, examples):
        for word in Text[0]:  
            pair = (word, lab)
            #freq[pair] = freq.get(pair,0)
            if pair in freq:
                freq[pair] += 1
            else:
                freq[pair] = 1
    return freq


###############################################################################
#extract positive and negative features
def extract_pos_neg_features(X,data_freq):
    '''
    Input: 
        X = Text file for which we need to extract number of positives and negatives
        data_freq = Frequence dictionary of occurrence of pos and neg labelled words
    '''
    sample = set(X) #Get unique vaues per dataset
    data_table = []
    for text in sample:
        pos = 0
        neg = 0
    
        if (text, 1) in data_freq:
            pos=data_freq[(text,1)]
    
        if (text, -1) in data_freq:
            neg = data_freq[(text,0)]
        
        data_table.append([text, pos, neg]) 
    
    PosValue=0
    NegValue=0
    for data in data_table:
        PosValue += data[1]
        NegValue += data[2]
    
    return [PosValue, NegValue]


###############################################################################
#Same as above
def extract_pos_neg_features1(X, freqs):
    
    '''This takes in a list containing the text and a word frequency dictionary
    For each unique word, it returns the number of occurrences with positive 
    and negative labels '''
    
    sample = set(X)
    x = np.zeros((1, 2)) 
  
    for text in sample:
        
        # increment the word count for both labels
        x[0,0] += freqs.get((text,1), 0)   
        x[0,1] += freqs.get((text,0), 0) 
    
    return x