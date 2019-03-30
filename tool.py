# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 12:12:13 2019

@author: admin
"""
import logging
import sys,os
from conlleval import return_report
import shutil
import torch.utils.data as data
import pickle
import torch

class Util():

    def __init__(self):
        pass


    def make_path(self, params):
        if not os.path.isdir(params.report_file):
            os.makedirs(params.report_file)
        if not os.path.isdir(params.ckpt_path):
            os.makedirs(params.ckpt_path)



    def get_logger(self, log_file):  # train.log
        # 1、创建一个logger
        logger = logging.getLogger('training.log')  # <Logger train.log (WARNING)>
        logger.setLevel(logging.DEBUG)  # 设置训练时的日志记录级别为debug级别
        # 2、创建一个handler，用于写入日志文件
        fh = logging.FileHandler(log_file) 
        fh.setLevel(logging.DEBUG)
        # 再创建一个handler，用于输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        # 3、定义handler的输出格式（formatter）
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        # 4、给handler添加formatter
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        # 5、给logger添加handler
        logger.addHandler(ch)
        logger.addHandler(fh)
        return logger





    def report_ner(self, results, output_file):
        """
        Run perl script to evaluate model
        """
        with open(output_file, "w", encoding='utf8') as f:
            to_write = []
            for block in results:
                for line in block:
                    to_write.append(line + "\n")
                to_write.append("\n")

            f.writelines(to_write)
        eval_lines = return_report(output_file)
        return eval_lines


    def result_to_json(self, string, tags):
        item = {"string": string, "entities": []}
        entity_name = ""
        entity_start = 0
        idx = 0
        for char, tag in zip(string, tags):
            if tag[0] == "S":
                item["entities"].append({"word": char, "start": idx, "end": idx + 1, "type": tag[2:]})
            elif tag[0] == "B":
                entity_name += char
                entity_start = idx
            elif tag[0] == "I":
                entity_name += char
            elif tag[0] == "E":
                entity_name += char
                item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": tag[2:]})
                entity_name = ""
            else:
                entity_name = ""
                entity_start = idx
            idx += 1
        return item
    def iobes_iob(self, tags):
        """
        IOBES -> IOB
        """
        new_tags = []
        for i, tag in enumerate(tags):
            if tag.split('-')[0] == 'B':
                new_tags.append(tag)
            elif tag.split('-')[0] == 'I':
                new_tags.append(tag)
            elif tag.split('-')[0] == 'S':
                new_tags.append(tag.replace('S-', 'B-'))
            elif tag.split('-')[0] == 'E':
                new_tags.append(tag.replace('E-', 'I-'))
            elif tag.split('-')[0] == 'O':
                new_tags.append(tag)
            else:
                raise Exception('Invalid format!')
        return new_tags
class sentence_data(data.Dataset):
    def __init__(self,data_path='dataMSRAtrain.pkl'):
        
        self.data_path=data_path
        with open(self.data_path,'rb') as f:
            self.word=pickle.load(f)
            self.word_f=pickle.load(f)
            self.word_s=pickle.load(f)
            self.tags=pickle.load(f)
              
    def __getitem__(self,idx):


        return torch.from_numpy(self.word[idx]).long(),torch.from_numpy(self.word_f[idx]).long(),torch.from_numpy(self.word_s[idx]).long(),torch.from_numpy(self.tags[idx])
    def __len__(self):
        return(len(self.word))




if __name__=='__main__':
    data_sentence=sentence_data()
    datal=data.DataLoader(data_sentence,batch_size=64,shuffle=False)
    print(len(datal))














