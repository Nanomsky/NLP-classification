#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 22:53:39 2020

@author: osita
"""

import nltk
import re
import string


string.punctuation
stopword = nltk.corpus.stopwords.words('english')


def clean_text(word):
    '''
    Removes punctuation
    Tokenize text into words
    Removes Stopwords
    ''' 
    no_punctuation = []
    for letter in word:
        if letter not in string.punctuation:
            no_punctuation.append(letter)
    no_punctuation = "".join(no_punctuation)
    
    text_tokenize = re.split('\W+', no_punctuation) # toknize by spliting on non word character
    text_cleaned = [word for word in text_tokenize if word not in stopword]
    
    return text_cleaned


def remove_stopwords(tokenized_list):
    '''
    Removes stopwords from a tokenized list 
    '''
    text = [word for word in tokenized_list if word not in stopword]
    return text

def tokenize(words):
    '''
    Tokenize a string of words
    '''
    tokens = re.split('\W+', words) # split on non word character
    return tokens

def remove_punction(word):
    '''
    Removes punctuation from a string of text
    '''
    no_punctuation = []
    for letter in word:
        if letter not in string.punctuation:
            no_punctuation.append(letter)
    no_punctuation = "".join(no_punctuation)
    
    return no_punctuation


def stemming(tokenized_text, Stemmer):
    '''
    Performs stemming with either Porter or Lancaster Stemmer
    '''
    if Stemmer == 'Porter':
        ps = nltk.PorterStemmer()

    elif Stemmer == 'Lancaster':
        ps = nltk.LancasterStemmer()

    text = [ps.stem(word) for word in tokenized_text]
    return text

def lemmatizing(tokenized_text):
    '''
    Applies the WordNetLemmatizer
    '''
    wn = nltk.WordNetLemmatizer()
    text = [wn.lemmatize(word) for word in tokenized_text]
    return text
