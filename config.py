# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:24:54 2019

@author: admin
"""
import sys,os

class Config:
      
      seg_dim  = 30      #切词信息维度
      char_dim = 300     #字向量模型维度
      lstm_dim = 400    #lstm 内部维度
      dropout  = 0.5
      learn_rate = 0.001  #学习率
      max_epoach  = 300    #最大训练次数
      batch_size = 64
      steps_check = 300   # 检查频率
      num_tags    = 14
      #num_chars   = 2641
     # num_segs    = 4     # 切词信息 四维   i b o e
     # filter_width = 3    # 卷积核大小
     # repeat_times = 4    # 膨胀卷积时卷积次数
      seed=333
      clip = 5
      optimizer = 'sgd'
      model_type = 'char_lstm_crf' # 训练模型
      tag_schema = 'iobes'
      pre_emb = True
      lower   = False
      zeros   = False
      clean   = False
      use_preembed=True
      
      momentum=0.99
      max_len=50
      max_leng=15
      root_path = os.getcwd() + os.sep
      # ckpt_path = os.path.join(root_path + 'ckpt', "")          # 模型路径
      #cnn_ckpt_path = os.path.join(root_path + 'ckpt\idcnn', '')
      ckpt_path = os.path.join(root_path + 'ckpt\char_lstm', '')
      log  = os.path.join(root_path + 'log')     # 训练日志记录
      train_file = os.path.join(root_path + 'dataMSRAtrain.pkl') # 训练数据集
      dev_file  = os.path.join(root_path + 'dataMRAval.pkl')  # 验证数据集
      test_file = os.path.join(root_path + 'textright1.txt') # 测试数据集
      report_file= os.path.join(root_path + 'result', 'ner_predict.utf8')  # 测试数据集
      word2vec=root_path+'msraword2vec.model'
      
      
      use_gpu=True
      
      
      
      id2t='id_to_tag.json'
      t2id='tag_to_id.json'
      id2char='id_to_char.json'
      char2id='char_to_id.json'
      
      word_pkl=root_path+'wordembed.pkl'
      assert  0 < dropout< 1, 'dropout must between 0, 1'
      assert  learn_rate > 0, 'learn_rate must > 0'
      assert  optimizer in ['adam', 'sgd', 'adagrad'] , 'this optimizer not exist'
