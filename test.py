# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 16:01:00 2019

@author: hasee
"""
import os
from gensim.models import word2vec
import logging
import jieba
import math
from string import punctuation
from heapq import nlargest


logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',level=logging.INFO)
path='C:\\Users\\hasee\\Documents\\news\\'
stopword_path='C:\\Users\\hasee\\Documents\\stopwords.txt'
output_path='C:\\Users\\hasee\\Documents\\news_abstract\\'


        
 
#################################################################################3
a='我爱中国共产党'
import jieba
b=list(jieba.cut(a))
c=list(jieba.cut(a,cut_all=True))
count={}
for i in c:
    ta=a.find(i)
    if ta not in count:
        count[ta]=[i]
    else:
        count[ta]+=[i]
s=set(a)
word2index={}
index1=[[] for _ in range(7)]




i=1
for j in s:
   
    word2index[j]=i
    i=i+1
for key,items in count.items():
    for i in range(len(count[key])):
        temp=[]
        str1=count[key][i]
        for word in str1:
            temp.append(word2index[word])
        count[key][i]=temp
print(count)    
for j in count:
    for i in   range(len(count[j])):
        index=len(count[j][i])+j-1
        index1[index].append(count[j][i])
print(index1)
        
        

import torch
x1=torch.tensor([[1,2,3],[2,3,4],[3,4,5]])
   
def kmax_pooling(x, dim, k):
    index = x.topk(k, dim = dim)[1].sort(dim = dim)[0]
    return x.gather(dim, index) 
print(x1.topk(2,1)[1].sort(1))

    
#################################################################
import torch as t
import time

class BasicModule(t.nn.Module):
    '''
    封装了nn.Module,主要是提供了save和load两个方法
    '''

    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name=str(type(self))# 默认名字

    def load(self, path,change_opt=True):
        print path
        data = t.load(path)
        if 'opt' in data:
            # old_opt_stats = self.opt.state_dict() 
            if change_opt:
                
                self.opt.parse(data['opt'],print_=False)
                self.opt.embedding_path=None
                self.__init__(self.opt)
            # self.opt.parse(old_opt_stats,print_=False)
            self.load_state_dict(data['d'])
        else:
            self.load_state_dict(data)
        return self.cuda()

    def save(self, name=None,new=False):
        prefix = 'checkpoints/' + self.model_name + '_' +self.opt.type_+'_'
        if name is None:
            name = time.strftime('%m%d_%H:%M:%S.pth')
        path = prefix+name

        if new:
            data = {'opt':self.opt.state_dict(),'d':self.state_dict()}
        else:
            data=self.state_dict()

        t.save(data, path)
        return path

    def get_optimizer(self,lr1,lr2=0,weight_decay = 0):
        ignored_params = list(map(id, self.encoder.parameters()))
        base_params = filter(lambda p: id(p) not in ignored_params,
                        self.parameters())
        if lr2 is None: lr2 = lr1*0.5 
        optimizer = t.optim.Adam([
                dict(params=base_params,weight_decay = weight_decay,lr=lr1),
                {'params': self.encoder.parameters(), 'lr': lr2}
            ])
        return optimizer
 
    # def save2(self,name=None):
    #     prefix = 'checkpoints/' + self.model_name + '_'
    #     if name is None:
    #         name = time.strftime('%m%d_%H:%M:%S.pth')
    #     path = prefix+name
    #     data = {'opt':self.opt.state_dict(),'d':self.state_dict()}
    #     t.save(data, path)
    #     return path
    # # def load2(self,path):
    # #     data = t.load(path)
    # #     self.__init__(data['opt'])
    # #     self.load_state_dict(data['d'])
from .BasicModule import BasicModule
import torch as t
import torch
import numpy as np
from torch import nn
from collections import OrderedDict
class FastText3(BasicModule):
    def __init__(self, opt ):
        super(FastText3, self).__init__()
        self.model_name = 'FastText3'
        self.opt=opt
        self.pre1 = nn.Sequential(
            nn.Linear(opt.embedding_dim,opt.embedding_dim*2),
            nn.BatchNorm1d(opt.embedding_dim*2),
            nn.ReLU(True)
        )

        self.pre2 = nn.Sequential(
            nn.Linear(opt.embedding_dim,opt.embedding_dim*2),
            nn.BatchNorm1d(opt.embedding_dim*2),
            nn.ReLU(True)
        )
        # self.pre_fc = nn.Linear(opt.embedding_dim,opt.embedding_dim*2)
        # self.bn = nn.BatchNorm1d(opt.embedding_dim*2)
        # self.pre_fc2 = nn.Linear(opt.embedding_dim,opt.embedding_dim*2)
        # self.bn2 = nn.BatchNorm1d(opt.embedding_dim*2) 

        self.encoder = nn.Embedding(opt.vocab_size,opt.embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(opt.embedding_dim*4,opt.linear_hidden_size),
            nn.BatchNorm1d(opt.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(opt.linear_hidden_size,opt.num_classes)
        )
        if opt.embedding_path:
            print('load embedding')
            self.encoder.weight.data.copy_(t.from_numpy(np.load(opt.embedding_path)['vector']))
 
    def forward(self,title,content):
        title_em = self.encoder(title)
        content_em = self.encoder(content)
        title_size = title_em.size()
        content_size = content_em.size()
        
 
    
        title_2 = self.pre1(title_em.contiguous().view(-1,256)).view(title_size[0],title_size[1],-1)
        content_2 = self.pre2(content_em.contiguous().view(-1,256)).view(content_size[0],content_size[1],-1)


        title_ = t.mean(title_2,dim=1)
        content_ = t.mean(content_2,dim=1)
        inputs=t.cat((title_.squeeze(),content_.squeeze()),1)
        out=self.fc(inputs)
        # content_out=self.content_fc(content.view(content.size(0),-1))
        # out=torch.cat((title_out,content_out),1)
        # out=self.fc(out)
        return out
#############################################################################
#padding and position wise
def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask 



############################################
#mask used for softmax
if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
########################################################       