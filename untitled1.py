# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 13:42:25 2019

@author: hasee
"""
class txt:
    def __init__(self):
        self.bn=3
        a=[]
        for i in range(10):
            
 
            setattr(self,'bn%i' %i,self.bn)
            a.append(self.bn)
        print(self.bn7)
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import jieba
co=['他们 今天','今天','很 高兴 去 玩','我','今天','约 他们 出去 玩']
corpus=[]










import os
import pickle
import numpy as np
import torch.nn as nn
import torch
from collections import OrderedDict
import torch.utils.data as data
import torch.nn.functional as F
path='D:\\NLP\\'
if not os.path.exists(path):
    os.makedirs(path)
with open (path+'word_embed.pkl','rb') as f:
   word_embed= pickle.load(f)
with open (path+'arr.pkl','rb') as f:
   arr= pickle.load(f)
with open(path+'word2index.pkl','rb') as f:
    word2index=pickle.load(f)
def mask_padding(arr):
    mask=np.zeros((arr.shape))
    mask[arr>0]=1
    return torch.from_numpy(mask).float()
mask=mask_padding(arr)
print(mask[0].size())
squelength=len(arr[0])
class sentence_data(data.Dataset):
    def __init__(self,data_path='D:\\NLP\\arr.pkl',mask=mask_padding):
        self.mask=mask_padding
        self.data_path=data_path
        with open(self.data_path,'rb') as f:
            arr=pickle.load(f)
        self.arr=arr      
    def __getitem__(self,idx):

        mask_l=self.mask(arr[idx])
        length=np.sum(arr[idx]>0)
        return torch.from_numpy(arr[idx]).long(),torch.from_numpy(arr[idx]).long(),mask_l,torch.tensor(length.item())
    def __len__(self):
        return(len(self.arr))
data_sentence=sentence_data()
datal=data.DataLoader(data_sentence,batch_size=5,shuffle=False)
'''
for x,y,z,e in datal:
    pass


print(x,y,z,e)
a,b=e.sort(descending=True)
print(a,b)
x1=x[b]
y1=y[b]
z1=z[b]
ls=torch.nn.LSTM(300,10,batch_first=True)
embed=nn.Embedding(len(word2index),300)
emb=embed(x1)

packed = torch.nn.utils.rnn.pack_padded_sequence(emb, a,batch_first=True)
lstm_out, r = ls(packed)
outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(lstm_out)
print(outputs.permute(1,0,2),output_lengths)
print(r[0].size())'''






'''

class Seq2SeqRNN(nn.Module):
    
    def __init__(self, rnn_type, input_size, embz_size, hidden_size, batch_size,output_size,
                 max_tgt_len, attention_type, tied_weight_type, pre_trained_vector, pre_trained_vector_type,
                 num_layers=1, encoder_drop=(0.2,0.3), decoder_drop=(0.2,0.3), 
                 bidirectional=True, bias=False, teacher_forcing=True):
        
        super().__init__()
        
        rnn_type, attention_type, tied_weight_type = rnn_type.upper(), attention_type.title(), tied_weight_type.lower()
        
        if rnn_type in ['LSTM', 'GRU']: self.rnn_type = rnn_type
        else: raise ValueError("""An invalid option for '--rnn_type' was supplied,
                                    options are ['LSTM', 'GRU']""")
            
        if attention_type in ['Luong', 'Bahdanau']: self.attention_type = attention_type
        else: raise ValueError("""An invalid option for '--attention_type' was supplied,
                                    options are ['Luong', 'Bahdanau']""")
            
        if tied_weight_type in ['three_way', 'two_way']: self.tied_weight_type = tied_weight_type
        else: raise ValueError("""An invalid option for '--tied_weight_type' was supplied,
                                    options are ['three_way', 'two_way']""")
    
                    
        #initialize model parameters    pre_train_vector:float tensor        
        self.output_size, self.embz_size, self.hidden_size = output_size, embz_size, hidden_size//2
        self.num_layers, self.input_size,  self.pre_trained_vector = num_layers, input_size,  pre_trained_vector
        self.bidirectional,self.teacher_forcing, self.pre_trained_vector_type = bidirectional, teacher_forcing, pre_trained_vector_type
        self.encoder_drop, self.decoder_drop = encoder_drop, decoder_drop
        
        
        if self.teacher_forcing: self.force_prob = 0.5
        
        #set bidirectional
        if self.bidirectional: self.num_directions = 2
        else: self.num_directions = 1
            
        
        #encoder0.2
        self.encoder_dropout = nn.Dropout(self.encoder_drop[0])
        #22435 300
        self.encoder_embedding_layer = nn.Embedding(self.input_size, self.embz_size)
        if self.pre_trained_vector: self.encoder_embedding_layer.weight.data.copy_(self.pre_trained_vector.weight.data)
         #300 200 bi drop 0.3   
        self.encoder_rnn = getattr(nn, self.rnn_type)(
                           input_size=self.embz_size,
                           hidden_size=self.hidden_size,
                           num_layers=self.num_layers,
                           dropout=self.encoder_drop[1], 
                           bidirectional=self.bidirectional,batch_first=True)
        # vector layer just for last hiden layer
        self.encoder_vector_layer = nn.Linear(self.hidden_size*self.num_directions,self.embz_size, bias=bias)
        
       #decoder
        self.decoder_dropout = nn.Dropout(self.decoder_drop[0])
        self.decoder_embedding_layer = nn.Embedding(self.input_size, self.embz_size)
        #bidrectional False
        self.decoder_rnn = getattr(nn, self.rnn_type)(
                           input_size=self.embz_size,
                           hidden_size=self.hidden_size*self.num_directions,
                           num_layers=self.num_layers,
                           dropout=self.decoder_drop[1],batch_first=True) 
        self.decoder_output_layer = nn.Linear(self.hidden_size*self.num_directions, self.embz_size, bias=bias)
        self.output_layer = nn.Linear(self.embz_size, self.output_size, bias=bias)
        
        #set tied weights: three way tied weights vs two way tied weights
        if self.tied_weight_type == 'three_way':
            self.decoder_embedding_layer.weight  = self.encoder_embedding_layer.weight
            self.output_layer.weight = self.decoder_embedding_layer.weight  
        else:
            if self.pre_trained_vector: self.decoder_embedding_layer.weight.data.copy_(self.pre_trained_vector.weight.data)
            self.output_layer.weight = self.decoder_embedding_layer.weight  
            
        #set attention 
        self.encoder_output_layer = nn.Linear(self.hidden_size*self.num_directions, self.embz_size, bias=bias)
        self.att_vector_layer = nn.Linear(self.embz_size+self.embz_size, self.embz_size,bias=bias)
        if self.attention_type == 'Bahdanau':
            self.decoder_hidden_layer = nn.Linear(self.hidden_size*self.num_directions, self.embz_size, bias=bias)
            self.att_score = nn.Linear(self.embz_size,1,bias=bias)

            
    
    def init_hidden(self, batch_size):
        if self.rnn_type == 'LSTM':
            return (torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size),
                    torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size))
        else:
            return torch.zeros(self.num_layers*self.num_directions, batch_size, self.hidden_size)
   
# cat the bidirectional hidden to unidirectional,so the hidden mast be (2,bacth,hidden)to(1,batch,2*hidden)
    def _cat_directions(self, hidden):
        def _cat(h):
            return torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
            
        if isinstance(hidden, tuple):
            # LSTM hidden contains a tuple (hidden state, cell state)
            hidden = tuple([_cat(h) for h in hidden])
        else:
            # GRU hidden
            hidden = _cat(hidden)
        return hidden    
    
# bahdanau_attention use the prehidden to  get the attention and cat the input    
    def bahdanau_attention(self, encoder_output, decoder_hidden, decoder_input):
        encoder_output = self.encoder_output_layer(encoder_output) 
        #batch sequencelength embedsize
        encoder_output = encoder_output
        decoder_hidden = decoder_hidden
        att_score = F.tanh(encoder_output + decoder_hidden)
        att_score = self.att_score(att_score)
        #batch sequencelength 1
        att_weight = F.softmax(att_score, dim=1)
        #bacth 1 sequencelength  *bacth sequencelength embedsize
        context_vector = torch.bmm(att_weight.transpose(-1, 1), encoder_output).squeeze(1)
        #bacth embedsize
        att_vector = torch.cat((context_vector, decoder_input), dim=1)
        #batch 2*embedsize
        att_vector = self.att_vector_layer(att_vector)
        #batch embedsize
        att_vector = F.tanh(att_vector)
        return att_weight.squeeze(-1), att_vector
        #batch sequencelength          batch embedsize
# luong_attention use the output of decoder to get the attention and cat the decoder output    
    def luong_attention(self, encoder_output, decoder_output):
        encoder_output = self.encoder_output_layer(encoder_output)
        #batch sequencelength embedsize
        encoder_output = encoder_output
        decoder_output = decoder_output
        #batch 1 embedsize
        att_score = torch.bmm(encoder_output, decoder_output.transpose(-1,1))
        #batch sequencelength 1
        att_weight = F.softmax(att_score, dim=1)
        context_vector = torch.bmm(att_weight.transpose(-1, 1), encoder_output).squeeze(1)
        #batch embedsize
        att_vector = torch.cat((context_vector, decoder_output.squeeze(1)), dim=1)
        #batch 2*embedsize
        att_vector = self.att_vector_layer(att_vector)
        #batch embedsize
        att_vector = F.tanh(att_vector)
        return att_weight.squeeze(-1), att_vector
        
    def decoder_forward(self, batch_size, encoder_output, decoder_hidden, y=None):
        decoder_input = torch.zeros(batch_size).long() 
        output_seq_stack, att_stack = [], []
        
        for i in range(self.max_tgt_len):
            decoder_input = self.decoder_dropout(self.decoder_embedding_layer(decoder_input))
            if self.attention_type == 'Bahdanau':
                if isinstance(decoder_hidden, tuple):
                    prev_hidden = self.decoder_hidden_layer(decoder_hidden[0][-1]).unsqueeze(0)
                else:
                    prev_hidden = self.decoder_hidden_layer(decoder_hidden[-1]).unsqueeze(0) 
                att, decoder_input = self.bahdanau_attention(encoder_output, prev_hidden, decoder_input)
                decoder_output, decoder_hidden = self.decoder_rnn(decoder_input.unsqueeze(0), decoder_hidden)
                decoder_output = self.decoder_output_layer(decoder_output.squeeze(0)) 
            else:
                decoder_output, decoder_hidden = self.decoder_rnn(decoder_input.unsqueeze(0), decoder_hidden)
                decoder_output = self.decoder_output_layer(decoder_output) 
                att, decoder_output = self.luong_attention(encoder_output, decoder_output)
            att_stack.append(att)
            output = self.output_layer(decoder_output)
            output_seq_stack.append(output)
            decoder_input = V(output.data.max(1)[1])
            if (decoder_input==1).all(): break 
            if self.teacher_forcing:    
                samp_prob = round(random.random(),1)
                if (y is not None) and (samp_prob < self.force_prob):
                    if i >= len(y): break
                    decoder_input = y[i] 
                
        return torch.stack(output_seq_stack), torch.stack(att_stack)
        
               
    def forward(self, seq, y=None):
        batch_size = seq[0].size(0)
        encoder_hidden = self.init_hidden(batch_size)
        encoder_input = self.encoder_dropout(self.encoder_embedding_layer(seq ))
        encoder_output, encoder_hidden = self.encoder_rnn(encoder_input, encoder_hidden) 
        if self.bidirectional:
            encoder_hidden = self._cat_directions(encoder_hidden)
        output = self.decoder_forward(batch_size, encoder_output, encoder_hidden, y=y)
#        if isinstance(encoder_hidden, tuple):
            encoder_vector = self.encoder_vector_layer(encoder_hidden[0][-1])
        else:
            encoder_vector = self.encoder_vector_layer(encoder_hidden[-1])
        output = output + (encoder_vector,)  
        return output
'''



