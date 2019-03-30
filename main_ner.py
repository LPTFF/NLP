# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 21:06:30 2019

@author: admin
"""
import torch
import torch.nn as nn
import numpy as np
import logging
import argparse
import json
import pickle
from config import Config
from tool import Util,sentence_data
from charlstm import Bilstm_CRF
import torch.optim as optim
import torch.utils.data as data
import time
import os
class Run(object):

    def __init__(self, train_path,val_path,log_file,tagset_size, gpu, n_char, n_embed, n_out,n_out1,pretrain_embed,check_points,max_epoch,clip,type='train',model='BILSTM_self_att_crf'):
        
        self.saver = None
        self.util = Util()
        self.train_loader=sentence_data(train_path)
        self.val_loader=sentence_data(val_path)
        self.model_type = model
        self.logger = self.util.get_logger(log_file)
        if self.model_type == 'BILSTM_self_att_crf':
            self.model = Bilstm_CRF(tagset_size, gpu, n_char, n_embed, n_out,n_out1,pretrain_embed)  
        else:
            raise 'model type error'
        self.ckpt_path = check_points
        if type == 'train':
            self.model=self.model.train()
        self.iobes_iob=self.util.iobes_iob
        self.max_epoch=max_epoch
        self.clip=clip
    def lr_decay(self,optimizer, epoch, decay_rate, init_lr):
        lr = init_lr * ((1-decay_rate)**epoch)
        print (" Learning rate is setted as:", lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer ,lr

    def save_model(self, model, epoch):
        
        torch.save(model.state_dict(), self.ckpt_path + '-' + str(epoch)+'.pth')
        self.logger.info('save model done')

    def save_best_model(self,model):
        torch.save(model.state_dict(),self.ckpt_path+'-'+'best_model.pth')

    def _data_preprocess(self,  char_to_id_path, id_to_char_path,tag_to_id_path,id_to_tag_path,train=True):
        if train:
            datal=data.DataLoader(self.train_loader,batch_size=64,shuffle=True)
        else:
            datal=data.DataLoader(self.val_loader,batch_size=1000,shuffle=False)
        
       # train_sentence = self.loader.load_sentences(data, zeros)
        print('数据总长度：{}'.format(len(datal)))
        with open (char_to_id_path,'r',encoding='utf-8') as f:
            char2id=json.load(f)
        with open(id_to_char_path,'r',encoding='utf-8') as f:
            id2char=json.load(f)
        with open(tag_to_id_path,'r',encoding='utf-8') as f:
            tag2id=json.load(f)
        with open(id_to_tag_path,'r',encoding='utf-8') as f:
            id2tag=json.load(f)
            
            
        #self.loader.update_tag_schema(train_sentence, self.config.tag_schema)
        #if sign:
            #mappings, char_to_id, id_to_char, tag_to_id, id_to_tag = self.loader.char_mapping(train_sentence,
                                                                                              #self.config.lower, sign)
            #train_data = self.loader.prepare_dataset(train_sentence, char_to_id, tag_to_id, self.config.lower)
       # else:
            #train_data = self.loader.prepare_dataset(train_sentence, char_to_id, tag_to_id, self.config.lower)

        #print('train 预处理后数据长度：{}'.format(len(train_data)))

        #batch_size = self.config.batch_size if sign else 100
        #batch_data = self.loader.batch_size_padding(train_data, batch_size)
        return datal,char2id,id2char,tag2id,id2tag

    def evaluate(self, model, data_manager, id_to_tag,id_to_word,report_file):
        ner_results = self._evaluate(model, data_manager, id_to_tag,id_to_word)
        report = self.util.report_ner(ner_results, report_file)
        return report

    def _evaluate(self,  model, data_manager, id_to_tag,id_to_word):
        """
        :param sess: session  to run the model
        :param data: list of data
        :param id_to_tag: index to tag name
        :return: evaluate result
        """
        results = []
        trans = model.eval()  # tensor.eval() 相当于 sess.run(self.trans)作用；其实就是执行
        for i,word,word_f,word_s,tags in enumerate(data_manager):
            x=(word,word_f,word_s,tags)
            loss,mask,best_path=trans(x)
            #batch_paths = self._decode(scores, lengths, trans)
            mask=mask.cpu()
            best_path=best_path.cpu()
            lengths=torch.sum(mask.long(),dim=1)
            assert len(mask) ==len(best_path)
            for i in range(len(best_path)):
                result = []
                string = [id_to_word[j.item()] for j in word[i][:lengths[i]]]
                gold = self.iobes_iob([id_to_tag[int(x.item())] for x in tags[i][:lengths[i]]])
                pred = self.iobes_iob([id_to_tag[int(x.item())] for x in best_path[i][:lengths[i]]])
                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([char, gold, pred]))
                results.append(result)
        return results



    #def _decode(self, logits, lengths, matrix):
        """
        :param logits: [batch_size, num_steps, num_tags]float32, logits
        :param lengths: [batch_size]int32, real length of each sequence
        :param matrix: transaction matrix for inference
        :return:
        """
      #  paths = []
       # small = -1000.0
       # start = np.asarray([[small] * self.config.num_tags + [0]])  # 初始化一个
        #for score, length in zip(logits, lengths):
           # score = score[:length]
           # pad = small * np.ones([length, 1])  # 创建一个字符长度是 输入字长度维度元素为1的np数组
           # logits = np.concatenate([score, pad], axis=1)
           # logits = np.concatenate([start, logits], axis=0)
           # path, _ = viterbi_decode(logits, matrix)
           # paths.append(path[1:])
       # return paths



   # def _run_sess(self, sess, batch, is_train):
      #  self._create_feed_dict(batch)
      #  if is_train:
       #     loss, train_op, lengths, trans, global_step, learn_rate = sess.run(
            #    [self.model.loss, self.model.train_op, self.model.lengths, self.model.trans, self.model.global_step,
            #     self.model.lr], self.feed_dict)
          #  return loss, lengths, trans, global_step, learn_rate
     #   else:
           # lengths, logits = sess.run([self.model.lengths, self.model.logits], self.feed_dict)
          #  return lengths, logits



   # def _create_feed_dict(self, batch, is_train=True):
     #   _, chars, segs, tags = batch
      #  self.feed_dict = {
       #     self.model.char_inputs: np.asarray(chars),
       #     self.model.seg_inputs: np.asarray(segs),
        #    self.model.dropout: 1.0,
      #  }
       # if is_train:
         #   self.feed_dict[self.model.targets] = np.asarray(tags)
          #  self.feed_dict[self.model.dropout] = self.config.dropout


    def _evaluate_line(self, model, inputs, id_to_tag,id_to_char):
        '''
        :param sess:
        :param inputs:
        :param id_to_tag:
        :return:
        '''
        trans = model.eval()
        loss,mask,best_path = trans(inputs)
        mask=mask.cpu()
        best_path=best_path.cpu()
        lengths=torch.sum(mask.long(),dim=1)
        leng=lengths[0]
        tags = [id_to_tag[idx.item()] for idx in best_path[0,:leng]]
        return self.util.result_to_json([id_to_char[j.item()] for j in inputs[0][0][:leng]], tags)


    def train(self,optimizer,char_to_id_path, id_to_char_path,tag_to_id_path,id_to_tag_path,learn_rate,moment,steps_check,report_file):

        trainloader,char2id,id2char,tag2id,id2tag=self._data_preprocess(  char_to_id_path, id_to_char_path,tag_to_id_path,id_to_tag_path,train=True)
        self.logger.info('train data prepare done')
        devloader, _, _, _ ,_= self._data_preprocess( char_to_id_path, id_to_char_path,tag_to_id_path,id_to_tag_path,train=False)
        self.logger.info('dev data prepare done')

        self.logger.info('start train......')
        batch_len = len(trainloader)

        init_lr=learn_rate
        if optimizer=='sgd':
            opt=torch.optim.SGD(self.model.parameters(),lr=learn_rate,momentum=moment)
        if optimizer=='adam':
            opt=torch.optim.Adam(self.model.parameters())
        if optimizer=='adagrad':
            opt=torch.optim.Adagrad(self.model.parameters())
        loss_data=1000
        global_step=0
        for epoch in range(self.max_epoch):

            steps=0   
            for i,[word,word_f,word_s,tags] in enumerate(trainloader):
                x=[word,word_f,word_s,tags]
                loss,mask,best_path=self.model(x)
            #batch_paths = self._decode(scores, lengths, trans)
            
            
            
                if (int(steps) + 1) % steps_check == 0:
                        self.logger.info(
                            ' epoch:{}, step/total_batch:{}/{}, global_step:{}, learn_rate:{}, loss:{}'.format(epoch,
                                                                                                               steps,
                                                                                                               batch_len,
                                                                                                              global_step,
                                                                                                             learn_rate,loss.data))               
                steps+=1
                global_step+=1                                                                              
                if (epoch + 1) % 2 == 0:
                    print('*' * 50)
                    with torch.no_grad():
                        report = self.evaluate(self.model.eval(), id2tag, id2char, report_file)
                        self.logger.info(report[1].strip())
                        self.logger.info('dev: epoch:{},  learn_rate:{}, loss:{}'.format(epoch, learn_rate, loss.data))
                    self.model.train()
                if (int(epoch) + 1) % 20 == 0:
                    self.save_model(self.model, epoch)
                    if loss.data<loss_data:
                        loss_data=loss.data
                        self.save_best_model(self.model)
                 
                torch.nn.utils.clip_grad_norm(self.model.parameters(),self.clip)    
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                opt, learn_rate=self.lr_decay(opt, epoch, 0.01, init_lr)
                



    def online(self, inputs,id_to_tag_path,id_to_char_path):
        if not inputs:
            return json.dumps({'result':'error'})

        with open(id_to_tag_path, 'r', encoding='utf-8') as tag, open(id_to_char_path,'r',encoding='utf-8') as char:
            id_to_tag = json.load(tag)
            
            id_to_char=json.load(char)
        
        with torch.no_grad():
            if os.path.isdir(self.ckpt_path):
                self.model=self.model.eval()
                self.model.load_state_dict(torch.load(self.ckpt_path+'-'+'best_model.pth'))
                
                
                
            
                self.logger.info('restore model')
                
                
                result = self._evaluate_line(self.model, inputs, id_to_tag,id_to_char)
                return json.dumps(result)
            else:
                raise 'not shuch file'




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tuning with muti_feature bi-directional LSTM-self_atten-CRF')
    parser.add_argument('--embedding',  help='Embedding for words', default=Config.word_pkl)
    parser.add_argument('--status', choices=['train', 'test', 'decode'], help='update algorithm', default='train')
    parser.add_argument('--savemodel', default=Config.ckpt_path)
    parser.add_argument('--savedset', help='Dir of saved data setting', default="data/save.dset")
    parser.add_argument('--train', default=Config.train_file) 
    parser.add_argument('--dev', default=Config.dev_file )  
    parser.add_argument('--test', default=Config.test_file) 
    parser.add_argument('--seg',help='add segement feature ', default=True) 
    parser.add_argument('--word-char',help='add word-char feature', default=True) 
    parser.add_argument('--Gpu',help='use gpu or not',default=True,type=bool) 
    parser.add_argument('--Optim',help='the optimizer',default=Config.optimizer)
    parser.add_argument('--LOG',help='logger path',default=Config.log) 
    parser.add_argument('--id2tag',default=Config.id2t)
    parser.add_argument('--tag2id',default=Config.t2id)
    parser.add_argument('--id2char',default=Config.id2char)
    parser.add_argument('--char2id',default=Config.char2id)
    
    parser.add_argument('--seg_dim',default=Config.seg_dim)
    parser.add_argument('--char_dim',default=Config.char_dim)
    parser.add_argument('--lstm_dim',default=Config.lstm_dim)
    parser.add_argument('--char_lstm_dim',default=200,type=int)
    parser.add_argument('--dropout',default=Config.dropout,type=float)
    parser.add_argument('--learn_rate',default=Config.learn_rate,type=float)
    parser.add_argument('--max_epoach',default=Config.max_epoach,type=int)
    parser.add_argument('--batch_size',default=Config.batch_size,type=int)
    parser.add_argument('--steps_check',default=Config.steps_check,type=int)
    parser.add_argument('--report_file',default=Config.report_file)
    parser.add_argument('--clip',help='clip model grad',default=Config.clip,type=float)
    parser.add_argument('--momentum',help='SGD momentum',default=Config.momentum,type=float)
    parser.add_argument('--max_len',help='The max length of the Lstm squence',default=Config.max_len,type=int)
    parser.add_argument('--max_leng',help='The max length of the char feature',default=Config.max_leng,type=int)   
    parser.add_argument('--num_tag',help='tag size',default=Config.num_tags,type=int)
    
    
    
    ut=Util()
    ut.make_path(Config)
    
    args = parser.parse_args()
    
    if not args.Gpu:
        if torch.cuda.is_available():
            print('you have a gpu ,you sure not to use it?')
    
       
    with open(args.embedding,'rb') as fp:
        embed= pickle.load(fp) 
    n_char=len(embed)
  


    run=Run (args.train,args.dev,args.LOG,args.num_tag, args.Gpu, n_char, args.char_dim, args.char_lstm_dim,args.lstm_dim,embed,args.savemodel,args.max_epoach,args.clip,type='train',model='BILSTM_self_att_crf')
    run.train(args.Optim,args.char2id, args.id2char,args.tag2id,args.id2tag,args.learn_rate,args.momentum,args.steps_check,args.report_file)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    