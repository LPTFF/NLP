# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 12:39:31 2019

@author: hasee
"""
import os
import pickle
import numpy as np
from gensim.models.keyedvectors import KeyedVectors

def check_parent_dir(path):
    """检查path的父目录是否存在，若不存在，则创建之
    Args:                      
        path: str, file path
    """
    
    if not os.path.exists(path):
        os.makedirs(path)


def object2pkl_file(path_pkl, ob):
    """将python对象写入pkl文件
    Args:
        path_pkl: str, pkl文件路径
        ob: python的list, dict, ...
    """
    with open(path_pkl, 'wb') as file_pkl:
        pickle.dump(ob, file_pkl)


def read_bin(path):
    """读取二进制文件
    Args:
        path: str, 二进制文件路径
    Returns:
        pkl_ob: pkl对象
    """
    file = open(path, 'rb')
    return pickle.load(file)
def load_embed_with_gensim(path_embed):
    """
    读取预训练的embedding
    Args:
        path_embed: str, bin or txt
    Returns:
        word_embed_dict: dict, 健: word, 值: np.array, vector
        word_dim: int, 词向量的维度
    """
    
    if path_embed.endswith('gz'):
        word_vectors = KeyedVectors.load_word2vec_format(path_embed, binary=True)
    elif path_embed.endswith('txt'):
        word_vectors = KeyedVectors.load_word2vec_format(path_embed, binary=False)
    else:
        raise ValueError('`path_embed` must be `bin` or `txt` file!')
    return word_vectors, word_vectors.vector_size


def prco_voc(stopword_path,train_path,path):
    check_parent_dir(path)
    with open(path+stopword_path,'r',encoding='utf8') as f:
        stopword=[]
        for i in f.readlines():
            stopword.append(i.strip())
    
    
    with open (path+train_path,'r',encoding='utf8') as fp:
        word2index={'pad':0,'unknown':1}
        a=[]
        length=[]
        sentence=[]
        for j in fp.readlines():
            b=[i for i in j.strip().split() if i not in stopword]
            a+=b
            length.append(len(b))
            sentence.append(b)
        voc=set(a)
        count=2
        for i in voc:
            if i not in word2index:
                word2index[i]=count
                count+=1
        arr = np.zeros((len(length),max(length)), dtype='int32')
        for ii in range(len(sentence)):
            for i in range(length[ii]):
                if sentence[ii][i] in voc:
                    arr[ii][i]=word2index[sentence[ii][i]]
                else:
                    arr[ii][i]=word2index['unknown']
            

    
    return word2index ,arr
            
            
            
    

path_embed='GoogleNews-vectors-negative300.bin.gz'           
def embedding_weight(path_embed,word2id_dict,seed=3):   
    assert path_embed.endswith('gz') or path_embed.endswith('txt')
    word2vec_model, word_dim = load_embed_with_gensim(path_embed)
    word_count = len(word2id_dict)   # 0 is for padding value
    np.random.seed(seed)
    scope = np.sqrt(3. / word_dim)
    word_embed_table = np.random.uniform(
        -scope, scope, size=(word_count, word_dim)).astype('float32')
    exact_match_count, fuzzy_match_count, unknown_count = 0, 0, 0
    for word in word2id_dict:
        if word in word2vec_model.vocab:
            word_embed_table[word2id_dict[word]] = word2vec_model[word]
            exact_match_count += 1
        elif word.lower() in word2vec_model.vocab:
            word_embed_table[word2id_dict[word]] = word2vec_model[word.lower()]
            fuzzy_match_count += 1
        else:
            unknown_count += 1
    total_count = exact_match_count + fuzzy_match_count + unknown_count
    return word_embed_table, exact_match_count, fuzzy_match_count, unknown_count, total_count
path='D:\\NLP\\'
stopword_path='stopword.txt'
train_path='train.txt'
word2index,arr= prco_voc(stopword_path,train_path,path)
word_embed_table,_,_,_,total_count=embedding_weight(path_embed,word2index)
object2pkl_file(path+'word_embed.pkl',word_embed_table)
object2pkl_file(path+'word2index.pkl',word2index)
object2pkl_file(path+'arr.pkl',arr)


















