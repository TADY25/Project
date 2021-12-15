
import tensorflow as tf
from tensorflow import keras

import spacy as spacy
from spacy.attrs import LOWER, POS, ENT_TYPE, IS_ALPHA
from spacy.tokens import Doc
import numpy as np
from tqdm import tqdm
from gensim.models import word2vec
from keras_funct import *
import os
import sys
import get_dataset
import pandas as pd
import numpy
import fr_core_news_md
from sklearn.feature_extraction.text import CountVectorizer



#Counts the type of each type of word in a sentence, our first feature. (This version is for the unlabelled data)
def type_counter_unlabelled(df_fit, df_transform):
    nlp = spacy.load('fr_core_news_md')
    dif_typ = []
    
    for i in range(df_fit.shape[0]):
        doc = nlp(df_fit['sentence'].iloc[i])
        
        for j in doc:
            if j.pos_ not in dif_typ:
                dif_typ.append(j.pos_)
                
    res = []
    
    for i in range(df_transform.shape[0]):
        temp = np.zeros(len(dif_typ))
        doc = nlp(df_transform['sentence'].iloc[i])
        for j in doc:
            ind = dif_typ.index(j.pos_)
            temp[ind] += 1
        res.append(temp)
        
                
    return res


#Also counts the types but for the training set
def type_counter(df):
    nlp = spacy.load('fr_core_news_md')
    dif_typ = []
    
    for i in range(df.shape[0]):
        doc = nlp(df['sentence'].iloc[i])
        for j in doc:
            if j.pos_ not in dif_typ:
                dif_typ.append(j.pos_)
                
    res = []
    
    for i in range(df.shape[0]):
        temp = np.zeros(len(dif_typ))
        doc = nlp(df['sentence'].iloc[i])
        for j in doc:
            ind = dif_typ.index(j.pos_)
            temp[ind] += 1
        res.append(temp)
        
                
    return res


def sk_vec(df):
    phrases = df['sentence'].tolist()
    vectorizer = CountVectorizer()
    vectorizer = CountVectorizer(ngram_range=(1, 2), analyzer='word' )
    return vectorizer.fit_transform(phrases).toarray()


#Same as before but fot the unlabelled data
def sk_vec_unlabelled(df_fit, df_transform):
    phrases_fit = df_fit['sentence'].tolist()
    phrases_transform = df_transform['sentence'].tolist()
    vectorizer = CountVectorizer(ngram_range=(1, 2), analyzer='word')
    vectorizer.fit(phrases_fit)
    return vectorizer.transform(phrases_transform).toarray()


def remove_tokens_on_match(doc):
    indexes = []
    for index, token in enumerate(doc):
        if (token.pos_  in ("PROPN", "PUNCT")):
            indexes.append(index)
    np_array = doc.to_array([LOWER, POS, ENT_TYPE, IS_ALPHA])
    np_array = numpy.delete(np_array, indexes, axis = 0)
    doc2 = Doc(doc.vocab, words=[t.text for i, t in enumerate(doc) if i not in indexes])
    doc2.from_array([LOWER, POS, ENT_TYPE, IS_ALPHA], np_array)
    return doc2


#This vectorizer is used
def vectorizer(df):
    tokenized_sentence = []
    nlp = spacy.load('fr_core_news_md')
    for sentence in df['sentence'].tolist():
        doc = nlp(sentence)
        doc = remove_tokens_on_match(doc)

        tokenized_sentence.append(doc.vector)
        
    return tokenized_sentence    


def word_len_counter(df):
    res = []
    nlp = spacy.load('fr_core_news_md')
    for sentence in df['sentence'].tolist():
        temp = []
        doc = nlp(sentence)
        
        temp.append(len(doc))
        temp.append(len(sentence))
        res.append(temp)
    return res
        
    

#Some data preparation to map difficulties into ints. Note that the we have y1 for standards classifiers and y2 for our neural classifier
def prepare_output(df):
    res1 = df['difficulty'].map({'A1':0, 'A2':1, 'B1':2,'B2':3,'C1':4,'C2':5}).tolist()
    
    res2 = []
    mapping =  {'A1':0, 'A2':1, 'B1':2,'B2':3,'C1':4,'C2':5}
    for i in range(len(df)):
        temp = np.zeros(6)
        temp[mapping[df['difficulty'].iloc[i]]] = 1
        res2.append(temp)
        
    
    return res1, res2


