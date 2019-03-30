# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 22:20:08 2019

@author: admin
"""
import collections
import re
import numpy as np
import torch
s='当/o ./o ,/o'
print(re.split('[,.。；！：？、》《（）\+//0-9‘’“”]/[o]',s))
chars=[1,1,1,2,3,3,3,3,3,3,3,3,4]
chars_list = collections.Counter(chars)
char_max = chars_list.most_common()
print('&gt;')
class cfg:
    _conv=3
    def __init__(self,**arg):
        print(self._conv)
        for i,j in arg.items():
            print(i,j)

import pickle
with open('wordembed.pkl','rb') as f:
    word=pickle.load(f)
print(word.shape)
hidden=torch.tensor([[[1,2,3],[2,3,4]],[[5,6,7],[3,4,5]]])
print(hidden.view(4,3).view(-1,2,3).contiguous())
a=getattr(torch.optim,'SGD')
print(a)
print(1-torch.eye(3))
b=torch.tensor([3])

print(x,a)
def to(a):
    a=a.cuda()
    print(a)
to(b)
print(b)