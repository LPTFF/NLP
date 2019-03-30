# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 13:24:19 2019

@author: admin
"""
from gensim.models import Word2Vec
from config import Config
import json
import numpy as np
import pickle
word2vec = Word2Vec.load('msraword2vec.model')
with open(Config.char2id,'r',encoding='utf8') as f:
    word2id_dict=json.load(f)

word_count = len(word2id_dict)   # 0 is for padding value
np.random.seed(Config.seed)
scope = np.sqrt(3. / Config.char_dim)
word_embed_table = np.random.uniform(
-scope, scope, size=(word_count, Config.char_dim)).astype('float32')
exact_match_count, fuzzy_match_count, unknown_count = 0, 0, 0
for word in word2id_dict:
    if word in word2vec:
       word_embed_table[word2id_dict[word]] = word2vec[word]
       
       exact_match_count += 1
    elif word.lower() in word2vec:
        word_embed_table[word2id_dict[word]] = word2vec[word.lower()]
        fuzzy_match_count += 1
    else:
        unknown_count += 1
total_count = exact_match_count + fuzzy_match_count + unknown_count
with open(Config.word_pkl,'wb') as fp:
    pickle.dump(word_embed_table,fp)
    
    
