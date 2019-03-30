# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 21:47:39 2019

@author: admin
"""
import numpy as np
from gensim.models import Word2Vec
import multiprocessing
output1='msraword2vec.model'
output2='msraword2vec.vector'
with open('wordtag.txt','r') as f:
    sentence=[]
    for lin in f.readlines():
        lin_list=[i.strip().split('/')[0] for i in lin.strip().split()]
        lines=''.join(lin_list)
        line=[l for l in lines]
        sentence.append(line)

model = Word2Vec(sentence, size=300, window=5, min_count=0, workers=multiprocessing.cpu_count())
model.save(output1)
model.wv.save_word2vec_format(output2, binary=False) 
