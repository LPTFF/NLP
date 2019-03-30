# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 22:08:17 2019

@author: admin
"""
import re
import pandas as pd
import numpy as np
import collections
import json
import jieba

BIO_T2D = {'' :0,
'B-ns' :1,
'B-nr' :2,
'B-nt' :3,
'I-ns' :4,
'I-nr' :5,
'I-nt' :6,
'O':7}

BIO_D2T = {0:'' ,
1:'B-ns' ,
2:'B-nr' ,
3:'B-nt' ,
4:'I-ns' ,
5:'I-nr' ,
6:'I_nt' ,
7: 'O'}
BIOES_T2D={'':0,'B-ns' :1,
'B-nr' :2,
'B-nt' :3,
'I-ns' :4,
'I-nr' :5,
'I-nt' :6,
'E-ns':7,
'E-nr':8,
'E-nt':9,
'S-nr':10,
'S-nt':11,
'S-ns':12,
'O':13}
BIOES_D2T={0:'',1:'B-ns' ,
2:'B-nr' ,
3:'B-nt' ,
4:'I-ns' ,
5:'I-nr' ,
6:'I-nt' ,
7:'E-ns',
8:'E-nr',
9:'E-nt',
10:'S-nr',
11:'S-nt',
12:'S-ns',
13:'O'}
#PAD_token=0
#SOS_token=1
#EOS_token=2
class processdata:
    def __init__(self,input_file='train1.txt',output_file='wordtag.txt',max_len=50,max_leng=15,BIOES=True,save_json=True,train=True):
        self.train=train
        self.save_json=save_json
        if BIOES:
            
            self.tag2id=BIOES_T2D
            self.id2tag=BIOES_D2T
        else:
            self.tag2id=BIO_T2D
            self.id2tag=BIO_D2T
            
        self.input_file=input_file
        self.output_file=output_file
        self.datas=[]
        self.labels=[]
        self.word2id={}
        self.id2word={}
        self.word2count={}
        self.max_len=max_len
        self.BIOES=BIOES
        self.max_leng=max_leng
# WORDTAG BIO    
    def wordtag(self):
        input_data = open(self.input_file,'r')
        output_data = open(self.output_file,'w')
        for line in input_data.readlines():
            
            line = line.strip().split()
            
            if len(line)==0:
                continue
            for word in line:
                word = word.split('/')
                if word[1]!='o':
                    if len(word[0])==1:
                        output_data.write(word[0]+"/B-"+word[1]+" ")
                    elif len(word[0])==2:
                        output_data.write(word[0][0]+"/B-"+word[1]+" ")
                        output_data.write(word[0][1]+"/I-"+word[1]+" ")
                    else:
                        output_data.write(word[0][0]+"/B-"+word[1]+" ")
                        for j in word[0][1:len(word[0])-1]:
                            output_data.write(j+"/I-"+word[1]+" ")
                        output_data.write(word[0][-1]+"/I-"+word[1]+" ")
                else:
                    for j in word[0]:
                        output_data.write(j+"/O"+" ")
            output_data.write('\n')
            
                
        input_data.close()
        output_data.close()
###################################################################################
    def BIO2BIOES(self, tags):
        """
        BIO -> IOBES
        """
        new_tags = []
        for i, tag in enumerate(tags):
            if tag == 'O':
                new_tags.append(tag)
            elif tag.split('-')[0] == 'B':  # 如果开头是 B
                if i + 1 != len(tags) and tags[i + 1].split('-')[0] == 'I':  # 如果下一个不是最后一个， 下一个是中间的；
                    new_tags.append(tag)  # 当下tag 放到新的tags里面
                else:
                    new_tags.append(tag.replace('B-', 'S-'))
            elif tag.split('-')[0] == 'I':
                if i + 1 < len(tags) and tags[i + 1].split('-')[0] == 'I':
                    new_tags.append(tag)
                else:
                    new_tags.append(tag.replace('I-', 'E-'))
            else:
                raise Exception('Invalid IOB format!')
        return new_tags    
######################################################################################
        

    def trim(self):
        self.wordtag()
        
            
        
        input_data = open(self.output_file,'r')
        for line in input_data.readlines():
            line=re.split('[,。；！：]/[o]',line.strip())
            for sen in line:
                sen = sen.strip().split()
                if len(sen)==0:
                    continue
                linedata=[]
                linelabel=[]
                num_not_o=0
                for word in sen:
                    word = word.split('/')
                    linedata.append(word[0])
                    linelabel.append(word[1])
        
                    if word[1]!='o':
                        num_not_o+=1
                if num_not_o!=0:
                    self.datas.append(linedata)
                    if self.BIOES:
                        linelabel=self.BIO2BIOES(linelabel)
                    self.labels.append(linelabel)
        
                        
        input_data.close()
        if self.train:
            chars=[]
            tags=[]
            for i in range(len(self.datas)):
                for j in range(len(self.datas[i])):
                    chars.append(self.datas[i][j])
                    tags.append(self.labels[i][j])
                    
            
            chars_list = collections.Counter(chars)   # 词数统计
            tags_list = collections.Counter(tags)     # tag统计
            print('char_list总长度：{}'.format(len(chars_list)))
            print('tag_list 总长度：{}'.format(len(tags_list)))
            char_max = chars_list.most_common()
            tags_max = tags_list.most_common()
    
            # char信息处理
            completion =  [('<PAD>', 10000001),('<UNK>', 10000000)]
            char_max = completion + char_max
            self.id_to_char = {i: v[0] for i, v in  enumerate(char_max)}    # {0: '<PAD>', 1: '<UNK>', 2: '0', 3: '，', 4: '：', 5: '。', 6: '无',}
            self.char_to_id = dict(zip(self.id_to_char.values(), self.id_to_char.keys()))
            # tag信息
            self.id_to_tag = {i: v[0] for i, v in enumerate(tags_max,1)}
            self.id_to_tag[0]=' '
            self.tag_to_id = dict(zip(self.id_to_tag.values(), self.id_to_tag.keys()))
            if self.save_json:
                with open('id_to_tag.json', 'w', encoding='utf-8') as fid2tag, \
                     open('tag_to_id.json', 'w', encoding='utf-8') as ftag2id, \
                     open('id_to_char.json', 'w', encoding='utf-8') as fid2char,\
                     open('char_to_id.json', 'w', encoding='utf-8') as fchar2id:
    
                         json.dump(self.id_to_tag,fid2tag,ensure_ascii = False)
                         json.dump(self.tag_to_id,ftag2id,ensure_ascii = False)
                         json.dump(self.id_to_char,fid2char,ensure_ascii = False)
                         json.dump(self.char_to_id,fchar2id,ensure_ascii = False)
        else:
            with open('id_to_tag.json', 'w', encoding='utf-8') as fid2tag, \
                     open('tag_to_id.json', 'w', encoding='utf-8') as ftag2id, \
                     open('id_to_char.json', 'w', encoding='utf-8') as fid2char,\
                     open('char_to_id.json', 'w', encoding='utf-8') as fchar2id:
    
                         self.id_to_tag=json.load(fid2tag)
                         self.tag_to_id=json.load(ftag2id)
                         self.id_to_char=json.load(fid2char)
                         self.char_to_id=json.load(fchar2id)
            
    def _get_seg_features(self, string,char_to_id):
        
        """
        Segment text with jieba
        features are represented in bies format
        s donates single word
        """
        word_feature=[]
        seg_feature = []
        
        for word in jieba.cut(string):
            
            
            if len(word) == 1:
                seg_feature.append(0)
                word_feature.append([char_to_id[word] if word in char_to_id else 1])
            else:
                tmp = [2] * len(word)
                tmp[0] = 1
                tmp[-1] = 3
                seg_feature.extend(tmp)
                for j in range(len(word)):
                    word_feature.append([char_to_id[word[i]] if word[i] in char_to_id else 1  for i in range(len(word))])
        return seg_feature,word_feature
    def mapping(self):
        self.trim()

        word=[]
        word_features=[]
        seg_features=[]
        tags=[]
        max_leng=self.max_leng
        for sen in self.datas:
            seg_feature,word_feature=self. _get_seg_features(''.join(sen),self.char_to_id)
            sen=[self.char_to_id[i] if i in self.char_to_id else 1 for i in sen]
            
            if len(sen)>=self.max_len:
                sen=sen[:self.max_len]
                seg_feature=seg_feature[:self.max_len]
                for i in range(self.max_len):
                    if len(word_feature[i])>=max_leng:
                        word_feature[i]=word_feature[i][:max_leng]
                    else:
                        word_feature[i]=word_feature[i]+[0]*(max_leng-len(word_feature[i]))
                word_feature=word_feature[:self.max_len]    
                
            else:
                sen.extend([0]*(self.max_len-len(sen)))
                seg_feature.extend([0]*(self.max_len-len(seg_feature)))
                for i in range(len(word_feature)):
                    if len(word_feature[i])>=max_leng: 
                        word_feature[i]=word_feature[i][:max_leng]
                    else:
                        word_feature[i]=word_feature[i]+[0]*(max_leng-len(word_feature[i]))
                        
                for j in range(self.max_len-len(word_feature)):
                    word_feature.append([1]*max_leng)
                
                
                
            word.append(sen) 
            word_features.append(word_feature)
            seg_features.append(seg_feature)
            
        for tag in self.labels:
            tag1=[self.tag_to_id[tag[i]] for i in range(len(tag))]
                
            
            if len(tag1)>=self.max_len:
                tag1=tag1[:self.max_len]
            else:
                tag1.extend([0]*(self.max_len-len(tag1)))
            tags.append(tag1)
        

        return word,word_features,seg_features,tags, self.id_to_tag, self.tag_to_id,self.id_to_char,self.char_to_id
            
          
        
p=processdata()
word,word_features,seg_features,tags,_,_,_,_=p.mapping()
word=np.array(word)
word_features=np.array(word_features)
seg_features=np.array(seg_features)

tags=np.array(tags)
print(word_features.shape,word.shape,tags.shape,seg_features.shape)
sample=np.random.permutation(np.arange(len(word)))
index=int(0.9*len(word))
word_tr=word[sample[:index]]
word_features_tr=word_features[sample[:index]]
seg_features_tr=seg_features[sample[:index]]
tags_tr=tags[sample[:index]]
word_te=word[sample[index:]]
word_features_te=word_features[sample[index:]]
seg_features_te=seg_features[sample[index:]]
tags_te=tags[sample[index:]]



print(word_tr[1],word_features_tr[1],seg_features_tr[1],tags_tr[1],len(word_tr[1]),len(word_features_tr[1]),len(seg_features_tr[1]),len(tags_tr[1]),len(word),len(word_features),len(seg_features),len(tags))



print ('Finished creating the data generator.')
import pickle
import os
with open('dataMSRAtrain.pkl', 'wb') as outp:
	pickle.dump(word_tr, outp)
	pickle.dump(word_features_tr, outp)
	pickle.dump(seg_features_tr, outp)
	pickle.dump(tags_tr, outp)
with open('dataMRAval.pkl','wb') as fp:
	
    pickle.dump(word_te, fp)
    pickle.dump(word_features_te, fp)
    pickle.dump(seg_features_te, fp)
    pickle.dump(tags_te,fp)
print ('Finished saving the data.')
