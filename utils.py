# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:23:02 2019

@author: hasee
"""
import os
import pickle


def check_parent_dir(path):
    """检查path的父目录是否存在，若不存在，则创建之
    Args:                      
        path: str, file path
    """
    parent_name = os.path.dirname(path)
    if not os.path.exists(parent_name):
        os.makedirs(parent_name)


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
#########################################################################
import numpy as np


def tokens2id_array(items, voc, oov_id=1):
    """
    将词序列映射为id序列
    Args:
        items: list, 词序列
        voc: item -> id的映射表
        oov_id: int, 未登录词的编号, default is 1
    Returns:
        arr: np.array, shape=[max_len,]
    """
    arr = np.zeros((len(items),), dtype='int32')
    for i in range(len(items)):
        if items[i] in voc:
            arr[i] = voc[items[i]]
        else:
            arr[i] = oov_id
    return arr
##################################################################################3
"""
用于从预训练词向量构建embedding表
"""





def build_word_embed(word2id_dict, path_embed, seed=137):
    """
    从预训练的文件中构建word embedding表
    Args:
        word2id_dict: dict, 健: word, 值: word id
        path_embed: str, 预训练的embedding文件，bin or txt
    Returns:
        word_embed_table: np.array, shape=[word_count, embed_dim]
        exact_match_count: int, 精确匹配的词数
        fuzzy_match_count: int, 精确匹配的词数
        unknown_count: int, 未匹配的词数
    """
    import numpy as np
    assert path_embed.endswith('bin') or path_embed.endswith('txt')
    word2vec_model, word_dim = load_embed_with_gensim(path_embed)
    word_count = len(word2id_dict) + 1  # 0 is for padding value
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
###################################################################################################################
def normalize_word(word):
    new_word = ''
    for c in word:
        if c.isdigit():
            new_word += '0'
        else:
            new_word += c
    return new_word
##############################################################################################################
import numpy as np
import torch
import torch.nn as nn


def init_cnn_weight(cnn_layer, seed=1337):
    """初始化cnn层权重
    Args:
        cnn_layer: weight.size() == [nb_filter, in_channels, [kernel_size]]
        seed: int
    """
    filter_nums = cnn_layer.weight.size(0)
    kernel_size = cnn_layer.weight.size()[2:]
    scope = np.sqrt(2. / (filter_nums * np.prod(kernel_size)))
    torch.manual_seed(seed)
    nn.init.normal_(cnn_layer.weight, -scope, scope)
    cnn_layer.bias.data.zero_()


def init_lstm_weight(lstm, num_layer=1, seed=1337):
    """初始化lstm权重
    Args:
        lstm: torch.nn.LSTM
        num_layer: int, lstm层数
        seed: int
    """
    for i in range(num_layer):
        weight_h = getattr(lstm, 'weight_hh_l{0}'.format(i))
        scope = np.sqrt(6.0 / (weight_h.size(0)/4. + weight_h.size(1)))
        torch.manual_seed(seed)
        nn.init.uniform_(getattr(lstm, 'weight_hh_l{0}'.format(i)), -scope, scope)

        weight_i = getattr(lstm, 'weight_ih_l{0}'.format(i))
        scope = np.sqrt(6.0 / (weight_i.size(0)/4. + weight_i.size(1)))
        torch.manual_seed(seed)
        nn.init.uniform_(getattr(lstm, 'weight_ih_l{0}'.format(i)), -scope, scope)

    if lstm.bias:
        for i in range(num_layer):
            weight_h = getattr(lstm, 'bias_hh_l{0}'.format(i))
            weight_h.data.zero_()
            weight_h.data[lstm.hidden_size: 2*lstm.hidden_size] = 1
            weight_i = getattr(lstm, 'bias_ih_l{0}'.format(i))
            weight_i.data.zero_()
            weight_i.data[lstm.hidden_size: 2*lstm.hidden_size] = 1


def init_linear(input_linear, seed=1337):
    """初始化全连接层权重
    """
    torch.manual_seed(seed)
    scope = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
    nn.init.uniform_(input_linear.weight, -scope, scope)
    if input_linear.bias is not None:
        input_linear.bias.data.zero_()


def init_embedding(input_embedding, seed=1337):
    """初始化embedding层权重
    """
    torch.manual_seed(seed)
    scope = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -scope, scope)
####################################################################
 #CRF layer   
    
    
    
    
    


    
