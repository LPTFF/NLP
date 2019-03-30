# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 21:02:17 2019

@author: admin
"""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np
'''we don't reset the hidden state of the lstm'''

class CharLSTM(nn.Module):

    def __init__(self, n_char, n_embed, n_out,pretrain_embed,use_pretrain=True,reset_lstm=True,max_len=50,use_cuda=True):
        super(CharLSTM, self).__init__()
        self.max_len=max_len
        
        self.cuda=use_cuda
        # the embedding layer
        self.embed = nn.Embedding(num_embeddings=n_char,
                                  embedding_dim=n_embed)
        # the lstm layer
        self.lstm = nn.LSTM(input_size=n_embed,
                            hidden_size=n_out // 2,
                            batch_first=True,
                            bidirectional=True)
        self.n_out=n_out
        if use_pretrain:
            self.embed.weight.data.copy_(torch.from_numpy(pretrain_embed))
        else:   
            self.reset_parameters()
        if reset_lstm:
            self.init_lstm_weight(self.lstm)
        if self.cuda:
            self.embed.cuda()
            self.lstm.cuda()
            
    def reset_parameters(self):
        bias = (3. / self.embed.weight.size(1)) ** 0.5
        nn.init.uniform_(self.embed.weight, -bias, bias)
    def init_lstm_weight(self,lstm, num_layer=1, seed=1337):
 
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


    
    
    def init_linear(self,input_linear, seed=1337):
    
        torch.manual_seed(seed)
        scope = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
        nn.init.uniform_(input_linear.weight, -scope, scope)
        if input_linear.bias is not None:
            input_linear.bias.data.zero_()

    
    
    
    
    def forward(self, x):
        if self.cuda:
            x=x.cuda()
        x=x.view(x.size(0)*self.max_len,-1).contiguous()
        mask = x.gt(0)
        lens, indices = torch.sort(mask.sum(dim=1), descending=True)
        _, inverse_indices = indices.sort()
        max_l = lens[0]
        x = x[indices, :max_l]
        x = self.embed(x)

        x = pack_padded_sequence(x, lens, True)
        x, (hidden, _) = self.lstm(x)
        reprs = torch.cat(torch.unbind(hidden), dim=1)
        reprs = reprs[inverse_indices]
        reprs=reprs.view(-1,self.max_len,self.n_out).contiguous()#batch squence hidden 64 50 300

        return reprs

class  WordLSTM(nn.Module):

    def __init__(self, n_char, n_embed, n_out,n_out1,pretrain_embed,use_pretrain=True,reset_lstm=True,reset_liear=True,max_len=50,dropout=0.5,seg_dim=30,seg_num=4,taget_size=14,drop=True,cuda=True):
        super(WordLSTM, self).__init__()
        self.max_len=max_len
        self.drop1=nn.Dropout(dropout)
        self.drop2=nn.Dropout(dropout)
        # the embedding layer
        self.embed1 = nn.Embedding(num_embeddings=n_char,
                                  embedding_dim=n_embed)
        #self.embed2=nn.Embedding(n_char,n_embed)
        self.embed3=nn.Embedding(seg_num,seg_dim)
        # the lstm layer
        self.hidden_dim=n_out1
        self.lstm = nn.LSTM(input_size=n_embed+n_out+seg_dim,
                            hidden_size=n_out1 // 2,
                            batch_first=True,
                            bidirectional=True)
        self.taget_size=taget_size+2
        self.n_out=n_out
        if use_pretrain:
            self.embed1.weight.data.copy_(torch.from_numpy(pretrain_embed))
           # self.embed2.weight.data.copy_(torch.from_numpy(pretrain_embed))
        else:   
            self.reset_parameters(self.embed1)
           # self.reset_parameters(self.embed2)
        self.reset_parameters(self.embed3)
        if reset_lstm:
            self.init_lstm_weight(self.lstm)
        self.hidden2tag = nn.Linear(n_out, self.taget_size)
        
        self.char_lstm=CharLSTM(n_char, n_embed, n_out,pretrain_embed)
        self.drop=drop
        self.att_lin1=nn.Linear(n_out1,n_out1)
        self.att_lin2=nn.Linear(n_out1,n_out1)
        self.att_tag=nn.Linear(n_out1+n_out1,self.taget_size)
        if reset_liear:
            self.init_linear(self.hidden2tag)
            self.init_linear(self.att_lin1)
            self.init_linear(self.att_lin2)
            self.init_linear(self.att_tag)
        if cuda:
            self.drop1.cuda()
            self.drop2.cuda()
            self.embed1.cuda()
            #self.embed2.cuda()
            self.embed3.cuda()
            self.lstm.cuda()
            self.hidden2tag.cuda()
            #self.char_lstm.cuda()
            self.att_lin1.cuda()
            self.att_lin2.cuda()
            self.att_tag.cuda()
    def reset_parameters(self,embed):
        bias = (3. / embed.weight.size(1)) ** 0.5
        nn.init.uniform_(embed.weight, -bias, bias)
        
        
    def init_lstm_weight(self,lstm, num_layer=1, seed=1337):
 
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

    def init_hidden(self,num):
        if self.cuda:
            return (torch.randn(2, num, self.hidden_dim // 2).cuda(),
                torch.randn(2, num, self.hidden_dim // 2).cuda())
        else:   
            return (torch.randn(2, num, self.hidden_dim // 2),
                torch.randn(2, num, self.hidden_dim // 2))
    
    def init_linear(self,input_linear, seed=1337):
    
        torch.manual_seed(seed)
        scope = np.sqrt(6.0 / (input_linear.weight.size(0) + input_linear.weight.size(1)))
        nn.init.uniform_(input_linear.weight, -scope, scope)
        if input_linear.bias is not None:
            input_linear.bias.data.zero_()
    def self_atten(self,input_tensor,mask):
        
        q=torch.nn.Tanh(self.att_lin1(input_tensor))
        
        k=torch.nn.Tanh(self.att_lin2(input_tensor))
        attn = torch.bmm(q,k.transpose(1,2))
        mask=mask.view(mask.size(0),mask.szie(1),1).expand_as(attn)
        attn=attn.masked_fill(mask, -np.inf)
        weight=torch.nn.Softmax(attn,dim=2)*(1-torch.eye(input_tensor.size(1)))
        att_out=torch.bmm(weight,input_tensor)
        cat_tensor=torch.cat((att_out,input_tensor),dim=2)
        return cat_tensor
        

    
    
    
    
    def forward(self, x):# x be a tuple (word,word_feature,word_segement,tags)
        
        
        
        if self.cuda:
            x[0]=x[0].cuda()
            x[1]=x[1].cuda()
            x[2]=x[2].cuda()
            x[3]=x[3].cuda()
        hidden_word_rnn=self.init_hidden(x[0].size(0))
        word_feature=x[1]
        rep=self.char_lstm(word_feature)
       
        mask2=x[0].gt(0)
        lens1,indices1=torch.sort(mask2.sum(dim=1),descending=True)
        
        _,inverse_indices1=indices1.sort()
        max_l1=lens1[0]
        word=self.embed1(x[0])
        word_seg=self.embed3(x[2])
        input_tensor=torch.cat([word,rep,word_seg],dim=2)
        input_tensor=self.drop1(input_tensor)
        input_tensor=input_tensor[indices1]
        input_tensor=pack_padded_sequence(input_tensor,lens1,True)
        lstm_out, hidden_word_rnn = self.lstm(input_tensor, hidden_word_rnn)
        lstm_out,x=nn.utils.rnn.pad_packed_sequence(lstm_out,batch_first=True)
        lstm_out=lstm_out[inverse_indices1]
        mask2=mask2[:,:max_l1]
        att_feats=self.self_atten(lstm_out,mask2)
        #lstm_feats = self.hidden2tag(lstm_out)
        lstm_feats=self.att_tag(att_feats)
        if self.drop:
            lstm_feats=self.drop2(lstm_feats)

        tags=x[3][:,:max_l1]
        
        
        
        return  lstm_feats, mask2, tags
    
    
 
    
    
    
    



 





   
########################################################################    


import torch.nn.functional as F

START_TAG = -2
STOP_TAG = -1


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec, m_size):
    """
    calculate log of exp sum
    args:
        vec (batch_size, vanishing_dim, hidden_dim) : input tensor
        m_size : hidden_dim
    return:
        batch_size, hidden_dim
    """
    _, idx = torch.max(vec, 1)  # B * 1 * M
    max_score = torch.gather(vec, 1, idx.view(-1, 1, m_size)).view(-1, 1, m_size)  # B * M
    return max_score.view(-1, m_size) + torch.log(torch.sum(torch.exp(vec - max_score.expand_as(vec)), 1)).view(-1, m_size)  # B * M

class CRF(nn.Module):

    def __init__(self, tagset_size, gpu):
        super(CRF, self).__init__()
        print ("build batched crf...")
        self.gpu = gpu
        # Matrix of transition parameters.  Entry i,j is the score of transitioning *to* i *from* j.
        self.average_batch = True
        self.tagset_size = tagset_size
        # # We add 2 here, because of START_TAG and STOP_TAG
        # # transitions (f_tag_size, t_tag_size), transition value from f_tag to t_tag
        init_transitions = torch.zeros(self.tagset_size+2, self.tagset_size+2)
        # init_transitions = torch.zeros(self.tagset_size+2, self.tagset_size+2)
        # init_transitions[:,START_TAG] = -1000.0
        # init_transitions[STOP_TAG,:] = -1000.0
        # init_transitions[:,0] = -1000.0
        # init_transitions[0,:] = -1000.0
        if self.gpu:
            init_transitions = init_transitions.cuda()
        self.transitions = nn.Parameter(init_transitions)

        # self.transitions = nn.Parameter(torch.Tensor(self.tagset_size+2, self.tagset_size+2))
        # self.transitions.data.zero_()

    def _calculate_PZ(self, feats, mask):
        """
            input:
                feats: (batch, seq_len, self.tag_size+2)
                masks: (batch, seq_len)
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        # print feats.view(seq_len, tag_size)
        assert(tag_size == self.tagset_size+2)
        mask = mask.transpose(1,0).contiguous()
        ins_num = seq_len * batch_size
        ## be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1)
        feats = feats.transpose(1,0).contiguous().view(ins_num,1, tag_size).expand(ins_num, tag_size, tag_size)
        ## need to consider start
        scores = feats + self.transitions.view(1,tag_size,tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)
        # build iter
        seq_iter = enumerate(scores)
        _, inivalues = next(seq_iter)  # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        partition = inivalues[:, START_TAG, :].clone().view(batch_size, tag_size, 1)  # bat_size * to_target_size

        ## add start score (from start to all tag, duplicate to batch_size)
        # partition = partition + self.transitions[START_TAG,:].view(1, tag_size, 1).expand(batch_size, tag_size, 1)
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: bat_size * from_target * to_target
            
            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            cur_partition = log_sum_exp(cur_values, tag_size)
            # print cur_partition.data
            
                # (bat_size * from_target * to_target) -> (bat_size * to_target)
            # partition = utils.switch(partition, cur_partition, mask[idx].view(bat_size, 1).expand(bat_size, self.tagset_size)).view(bat_size, -1)
            mask_idx = mask[idx, :].view(batch_size, 1).expand(batch_size, tag_size)
            
            ## effective updated partition part, only keep the partition value of mask value = 1
            masked_cur_partition = cur_partition.masked_select(mask_idx)
            ## let mask_idx broadcastable, to disable warning
            mask_idx = mask_idx.contiguous().view(batch_size, tag_size, 1)

            ## replace the partition where the maskvalue=1, other partition value keeps the same
            partition.masked_scatter_(mask_idx, masked_cur_partition)  
        # until the last state, add transition score for all partition (and do log_sum_exp) then select the value in STOP_TAG
        cur_values = self.transitions.view(1,tag_size, tag_size).expand(batch_size, tag_size, tag_size) + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
        cur_partition = log_sum_exp(cur_values, tag_size)
        final_partition = cur_partition[:, STOP_TAG]
        return final_partition.sum(), scores


    def _viterbi_decode(self, feats, mask):
        """
            input:
                feats: (batch, seq_len, self.tag_size+2)
                mask: (batch, seq_len)
            output:
                decode_idx: (batch, seq_len) decoded sequence
                path_score: (batch, 1) corresponding score for each sequence (to be implementated)
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        assert(tag_size == self.tagset_size+2)
        ## calculate sentence length for each sentence
        length_mask = torch.sum(mask.long(), dim = 1).view(batch_size,1).long()
        ## mask to (seq_len, batch_size)
        mask = mask.transpose(1,0).contiguous()
        ins_num = seq_len * batch_size
        ## be careful the view shape, it is .view(ins_num, 1, tag_size) but not .view(ins_num, tag_size, 1)
        feats = feats.transpose(1,0).contiguous().view(ins_num, 1, tag_size).expand(ins_num, tag_size, tag_size)
        ## need to consider start
        scores = feats + self.transitions.view(1,tag_size,tag_size).expand(ins_num, tag_size, tag_size)
        scores = scores.view(seq_len, batch_size, tag_size, tag_size)

        # build iter
        seq_iter = enumerate(scores)
        ## record the position of best score
        back_points = list()
        partition_history = list()
        
        
        ##  reverse mask (bug for mask = 1- mask, use this as alternative choice)
        # mask = 1 + (-1)*mask
        mask =  (1 - mask.long()).byte()
        _, inivalues = next(seq_iter) # bat_size * from_target_size * to_target_size
        # only need start from start_tag
        partition = inivalues[:, START_TAG, :].clone().view(batch_size, tag_size, 1)  # bat_size * to_target_size
        partition_history.append(partition)
        # iter over last scores
        for idx, cur_values in seq_iter:
            # previous to_target is current from_target
            # partition: previous results log(exp(from_target)), #(batch_size * from_target)
            # cur_values: batch_size * from_target * to_target
            cur_values = cur_values + partition.contiguous().view(batch_size, tag_size, 1).expand(batch_size, tag_size, tag_size)
            ## forscores, cur_bp = torch.max(cur_values[:,:-2,:], 1) # do not consider START_TAG/STOP_TAG
            partition, cur_bp = torch.max(cur_values, 1)
            partition_history.append(partition)
            ## cur_bp: (batch_size, tag_size) max source score position in current tag
            ## set padded label as 0, which will be filtered in post processing
            cur_bp.masked_fill_(mask[idx].view(batch_size, 1).expand(batch_size, tag_size), 0) 
            back_points.append(cur_bp)
        ### add score to final STOP_TAG
        partition_history = torch.cat(partition_history,0).view(seq_len, batch_size,-1).transpose(1,0).contiguous() ## (batch_size, seq_len. tag_size)
        ### get the last position for each setences, and select the last partitions using gather()
        last_position = length_mask.view(batch_size,1,1).expand(batch_size, 1, tag_size) -1
        last_partition = torch.gather(partition_history, 1, last_position).view(batch_size,tag_size,1)
        ### calculate the score from last partition to end state (and then select the STOP_TAG from it)
        last_values = last_partition.expand(batch_size, tag_size, tag_size) + self.transitions.view(1,tag_size, tag_size).expand(batch_size, tag_size, tag_size)
        _, last_bp = torch.max(last_values, 1)
        pad_zero = torch.zeros(batch_size, tag_size).long()
        if self.gpu:
            pad_zero = pad_zero.cuda()
        back_points.append(pad_zero)
        back_points  =  torch.cat(back_points).view(seq_len, batch_size, tag_size)
        
        ## select end ids in STOP_TAG
        pointer = last_bp[:, STOP_TAG]
        insert_last = pointer.contiguous().view(batch_size,1,1).expand(batch_size,1, tag_size)
        back_points = back_points.transpose(1,0).contiguous()
        ## move the end ids(expand to tag_size) to the corresponding position of back_points to replace the 0 values
        # print "lp:",last_position
        # print "il:",insert_last
        back_points.scatter_(1, last_position, insert_last)
        # print "bp:",back_points
        # exit(0)
        back_points = back_points.transpose(1,0).contiguous()
        ## decode from the end, padded position ids are 0, which will be filtered if following evaluation
        decode_idx = torch.LongTensor(seq_len, batch_size)
        if self.gpu:
            decode_idx = decode_idx.cuda()
        decode_idx[-1] = pointer.data
        for idx in range(len(back_points)-2, -1, -1):
            pointer = torch.gather(back_points[idx], 1, pointer.contiguous().view(batch_size, 1))
            decode_idx[idx] = pointer.data
        path_score = None
        decode_idx = decode_idx.transpose(1,0)
        return path_score, decode_idx



    def forward(self, feats,mask):
    	path_score, best_path = self._viterbi_decode(feats,mask)
    	return path_score, best_path
        

    def _score_sentence(self, scores, mask, tags):
        """
            input:
                scores: variable (seq_len, batch, tag_size, tag_size)
                mask: (batch, seq_len)
                tags: tensor  (batch, seq_len)
            output:
                score: sum of score for gold sequences within whole batch
        """
        # Gives the score of a provided tag sequence
        batch_size = scores.size(1)
        seq_len = scores.size(0)
        tag_size = scores.size(2)
        ## convert tag value into a new format, recorded label bigram information to index  
        new_tags = torch.LongTensor(batch_size, seq_len)
        if self.gpu:
            new_tags = new_tags.cuda()
        for idx in range(seq_len):
            if idx == 0:
                ## start -> first score
                new_tags[:,0] =  (tag_size - 2)*tag_size + tags[:,0]

            else:
                new_tags[:,idx] =  tags[:,idx-1]*tag_size + tags[:,idx]

        ## transition for label to STOP_TAG
        end_transition = self.transitions[:,STOP_TAG].contiguous().view(1, tag_size).expand(batch_size, tag_size)
        ## length for batch,  last word position = length - 1
        length_mask = torch.sum(mask.long(), dim = 1).view(batch_size,1).long()
        ## index the label id of last word
        end_ids = torch.gather(tags, 1, length_mask - 1)

        ## index the transition score for end_id to STOP_TAG
        end_energy = torch.gather(end_transition, 1, end_ids)

        ## convert tag as (seq_len, batch_size, 1)
        new_tags = new_tags.transpose(1,0).contiguous().view(seq_len, batch_size, 1)
        ### need convert tags id to search from 400 positions of scores
        tg_energy = torch.gather(scores.view(seq_len, batch_size, -1), 2, new_tags).view(seq_len, batch_size)  # seq_len * bat_size
        ## mask transpose to (seq_len, batch_size)
        tg_energy = tg_energy.masked_select(mask.transpose(1,0))
        
        # ## calculate the score from START_TAG to first label
        # start_transition = self.transitions[START_TAG,:].view(1, tag_size).expand(batch_size, tag_size)
        # start_energy = torch.gather(start_transition, 1, tags[0,:])

        ## add all score together
        # gold_score = start_energy.sum() + tg_energy.sum() + end_energy.sum()
        gold_score = tg_energy.sum() + end_energy.sum()
        return gold_score

    def neg_log_likelihood_loss(self, feats, mask, tags):
        # nonegative log likelihood
        batch_size = feats.size(0)
        forward_score, scores = self._calculate_PZ(feats, mask)
        gold_score = self._score_sentence(scores, mask, tags)
        # print "batch, f:", forward_score.data[0], " g:", gold_score.data[0], " dis:", forward_score.data[0] - gold_score.data[0]
        # exit(0)
        if self.average_batch:
            return (forward_score - gold_score)/batch_size
        else:
             return forward_score - gold_score
    
    
    
    
class Bilstm_CRF(nn.Module):

    def __init__(self, tagset_size, gpu, n_char, n_embed, n_out,n_out1,pretrain_embed):
        super(Bilstm_CRF, self).__init__()
        self.tagset_size=tagset_size
        self.use_gpu=gpu
        self.n_char=n_char
        self.n_embed=n_embed
        self.n_out=n_out
        self.n_out1=n_out1
        self.pretrain_embed=pretrain_embed
        self.word_lstm=WordLSTM(self.n_char,self.n_embed,self.n_out,self.n_out1,self.pretrain_embed)
        self.crf=CRF(self.tagset_size,self.use_gpu)
    def forward(self,x):
        feats,mask,tags=self.word_lstm(x)
        loss=self.crf.neg_log_likelihood_loss(feats,mask,tags)
        _,best_path= self.crf(feats,mask)
        return loss,mask,best_path
        
        
                
        
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

















   