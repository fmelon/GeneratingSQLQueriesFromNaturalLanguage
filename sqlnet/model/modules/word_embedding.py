import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class WordEmbedding(nn.Module):
    def __init__(self, word_emb, N_word, gpu, SQL_TOK,
            our_model, trainable=False):
        super(WordEmbedding, self).__init__()
        self.trainable = trainable
        self.N_word = N_word
        self.our_model = our_model
        self.gpu = gpu
        self.SQL_TOK = SQL_TOK

        if trainable:
            print "Using trainable embedding"
            self.w2i, word_emb_val = word_emb
            self.embedding = nn.Embedding(len(self.w2i), N_word)
            self.embedding.weight = nn.Parameter(
                    torch.from_numpy(word_emb_val.astype(np.float32)))
        else:
            self.word_emb = word_emb
            print "Using fixed embedding"


    def gen_x_batch(self, q, col):
        B = len(q)
        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        for i, (single_q, single_col) in enumerate(zip(q, col)):
            if self.trainable:
                q_val = map(lambda x:self.w2i.get(x, 0), single_q)
            else:
                q_val = map(lambda x:self.word_emb.get(x, np.zeros(self.N_word, dtype=np.float32)), single_q)
            if self.our_model:
                if self.trainable:
                    val_embs.append([1] + q_val + [2])  #<BEG> and <END>
                else:
                    val_embs.append([np.zeros(self.N_word, dtype=np.float32)] + q_val + [np.zeros(self.N_word, dtype=np.float32)])  #<BEG> and <END>
                val_len[i] = 1 + len(q_val) + 1
            else:
                single_col_all = [x for toks in single_col for x in toks+[',']]
                if self.trainable:
                    col_val = map(lambda x:self.w2i.get(x, 0), single_col_all)
                    val_embs.append( [0 for _ in self.SQL_TOK] + col_val + [0] + q_val+ [0])
                else:
                    col_val = map(lambda x:self.word_emb.get(x, np.zeros(self.N_word, dtype=np.float32)), single_col_all)
                    val_embs.append( [np.zeros(self.N_word, dtype=np.float32) for _ in self.SQL_TOK] + col_val + [np.zeros(self.N_word, dtype=np.float32)] + q_val+ [np.zeros(self.N_word, dtype=np.float32)])
                val_len[i] = len(self.SQL_TOK) + len(col_val) + 1 + len(q_val) + 1
        max_len = max(val_len)

        if self.trainable:
            val_tok_array = np.zeros((B, max_len), dtype=np.int64)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_tok_array[i,t] = val_embs[i][t]
            val_tok = torch.from_numpy(val_tok_array)
            if self.gpu:
                val_tok = val_tok.cuda()
            val_tok_var = Variable(val_tok)
            val_inp_var = self.embedding(val_tok_var)
        else:
            val_emb_array = np.zeros((B, max_len, self.N_word), dtype=np.float32)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_emb_array[i,t,:] = val_embs[i][t]
            val_inp = torch.from_numpy(val_emb_array)
            if self.gpu:
                val_inp = val_inp.cuda()
            val_inp_var = Variable(val_inp)
        return val_inp_var, val_len

    def gen_col_batch(self, cols):
        ret = []
        col_len = np.zeros(len(cols), dtype=np.int64)

        names = []
        for b, single_cols in enumerate(cols):
            names = names + single_cols
            col_len[b] = len(single_cols)

        name_inp_var, name_len = self.str_list_to_batch(names)
        return name_inp_var, name_len, col_len

    def str_list_to_batch(self, str_list):
        B = len(str_list)

        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        for i, single_str in enumerate(str_list):
            if self.trainable:
                val = [self.w2i.get(x, 0) for x in single_str]
            else:
                val = [self.word_emb.get(x, np.zeros(
                    self.N_word, dtype=np.float32)) for x in single_str]
            val_embs.append(val)
            val_len[i] = len(val)
        max_len = max(val_len)

        if self.trainable:
            val_tok_array = np.zeros((B, max_len), dtype=np.int64)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_tok_array[i,t] = val_embs[i][t]
            val_tok = torch.from_numpy(val_tok_array)
            if self.gpu:
                val_tok = val_tok.cuda()
            val_tok_var = Variable(val_tok)
            val_inp_var = self.embedding(val_tok_var)
        else:
            val_emb_array = np.zeros(
                    (B, max_len, self.N_word), dtype=np.float32)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_emb_array[i,t,:] = val_embs[i][t]
            val_inp = torch.from_numpy(val_emb_array)
            if self.gpu:
                val_inp = val_inp.cuda()
            val_inp_var = Variable(val_inp)

        return val_inp_var, val_len


class CharacterEmbedding(nn.Module):
    def __init__(self, word_emb,character_emb, N_word, gpu, SQL_TOK,
            our_model, trainable=False):
        super(CharacterEmbedding, self).__init__()
        self.trainable = trainable
        self.N_word = N_word
        self.our_model = our_model
        self.SQL_TOK = SQL_TOK
        self.gpu = gpu
        
	if not trainable:
              self.character_emb = character_emb
              self.word_emb = word_emb  
        else:
            print "Using trainable embedding"
            self.w2i, word_emb_val = word_emb
            self.embedding = nn.Embedding(len(self.w2i), N_word)
            self.embedding.weight = nn.Parameter(
                  torch.from_numpy(word_emb_val.astype(np.float32)))
            self.character_emb = character_emb

        channel_output = 100
        channel_input = 300
 
        if character_emb != None:
            self.c2i, character_emb_val = character_emb

            self.character_embedding = nn.Embedding(len(self.c2i), N_word)
            self.character_embedding.weight = nn.Parameter(
                    torch.from_numpy(character_emb_val.astype(np.float32)))

            self.unit1_cnn, self.unit2_cnn, self.unit3_cnn, self.dropout = self.get_CNNS(channel_input, channel_output)


    def get_CNNS(self, channel_input, channel_output):

        unit3_cnn = nn.Conv2d(channel_input, channel_output, (1, 3))
        unit1_cnn = nn.Conv2d(channel_input, channel_output, (1, 5))
        unit2_cnn = nn.Conv2d(channel_input, channel_output, (1, 4))
        dropout = nn.Dropout(0.3)

	return unit1_cnn, unit2_cnn, unit3_cnn, dropout


    
    def gen_x_batch(self, question, col):
        B = len(question)
        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        for i, (single_question, single_col) in enumerate(zip(question, col)):
            if self.trainable:
                question_val = map(lambda x:self.w2i.get(x, 0), single_question)
            else:
                question_val = map(lambda x:self.word_emb.get(x, np.zeros(self.N_word, dtype=np.float32)), single_question)


            if self.our_model:
                if self.trainable:
                    val_embs.append([1] + question_val + [2])  #<BEG> and <END>
                else:
                    val_embs.append([np.zeros(self.N_word, dtype=np.float32)] + question_val + [np.zeros(self.N_word, dtype=np.float32)])  #<BEG> and <END>
                val_len[i] = 1 + len(question_val) + 1
            else:
                single_col_all = [x for toks in single_col for x in toks+[',']]
                if self.trainable:
                    col_val = map(lambda x:self.w2i.get(x, 0), single_col_all)
                    val_embs.append( [0 for _ in self.SQL_TOK] + col_val + [0] + question_val+ [0])
                else:
                    col_val = map(lambda x:self.word_emb.get(x, np.zeros(self.N_word, dtype=np.float32)), single_col_all)
                    val_embs.append( [np.zeros(self.N_word, dtype=np.float32) for _ in self.SQL_TOK] + col_val + [np.zeros(self.N_word, dtype=np.float32)] + question_val+ [np.zeros(self.N_word, dtype=np.float32)])
                val_len[i] = len(self.SQL_TOK) + len(col_val) + 1 + len(question_val) + 1
        max_len = max(val_len)

        if self.trainable:
            val_tok_array = np.zeros((B, max_len), dtype=np.int64)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_tok_array[i,t] = val_embs[i][t]
            val_tok = torch.from_numpy(val_tok_array)
            if self.gpu:
                val_tok = val_tok.cuda()
            val_tok_var = Variable(val_tok)
            val_inp_var = self.embedding(val_tok_var)
        else:
            val_emb_array = np.zeros((B, max_len, self.N_word), dtype=np.float32)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_emb_array[i,t,:] = val_embs[i][t]
            val_inp = torch.from_numpy(val_emb_array)
            if self.gpu:
                val_inp = val_inp.cuda()
            val_inp_var = Variable(val_inp)

        if self.character_emb != None:
            B = len(question)

            question_character = []
            max_character_len = 0
            for single_text in question:
                single_text_res = []
                for word in single_text:
                    cnt = 0
                    single_word_res = []
                    for c in word:
                        cnt+=1
                        single_word_res.append(self.c2i[c])
                        if cnt>max_character_len:
                            max_character_len = cnt
                    single_text_res.append(single_word_res)
                question_character.append(single_text_res)


            col_character = []
            for single_col_list in col:
                single_col_list_res = []
                for single_col in single_col_list:
                    single_col_res = []
                    for word in single_col:
                        single_word_res = []
                        for c in word:
                           single_word_res.append(self.c2i[c])
                        single_col_res.append(single_word_res)
                    single_col_list_res.append(single_col_res)
                col_character.append(single_col_list_res)



	    temp = self.get_temp_emb(question_character, B, max_len, max_character_len)

            temp2 = torch.from_numpy(temp)

            if self.gpu:
                temp2 = temp2.cuda()
            temp3 = Variable(temp2)
            temp3 = temp3.view(B,-1)
            character_emb_temp = self.character_embedding(temp3)
            character_emb_temp = character_emb_temp.view(B, max_len, max_character_len,self.N_word).permute(0,3,1,2)

           
            character_res1_maxperm, character_res2_maxperm, character_res3_maxperm = self.get_max_perm(character_emb_temp)

            character_res_final = torch.cat([character_res1_maxperm,character_res2_maxperm,character_res3_maxperm],-1)

            character_res_final = self.dropout(character_res_final)

            val_inp_var = torch.cat([character_res_final,val_inp_var],-1)

        return val_inp_var, val_len

    def get_temp_emb(self, question_character, B, max_len, max_character_len):

        emb_arr = np.zeros((B, max_len, max_character_len), dtype=np.int64)

        for i in range(B):
            for j in range(len(question_character[i])):
                for k in range(len(question_character[i][j])):
                    emb_arr[i, j, k] = question_character[i][j][k]
  
	return emb_arr

    def get_max_perm(self, character_emb_temp):

        character_res1 = F.relu(self.unit1_cnn(character_emb_temp))
        character_res2 = F.relu(self.unit2_cnn(character_emb_temp))
        character_res3 = F.relu(self.unit3_cnn(character_emb_temp))

        character_res1_maxperm = character_res1.max(3)[0].permute(0,2,1)
        character_res2_maxperm = character_res2.max(3)[0].permute(0,2,1)
        character_res3_maxperm = character_res3.max(3)[0].permute(0,2,1)

        return character_res1_maxperm, character_res2_maxperm, character_res3_maxperm


    def gen_col_batch(self, cols):
        ret = []
        col_len = np.zeros(len(cols), dtype=np.int64)

        names = []
        for b, single_cols in enumerate(cols):
            names = names + single_cols
            col_len[b] = len(single_cols)

        name_inp_var, name_len = self.str_list_to_batch(names)
        return name_inp_var, name_len, col_len

    def str_list_to_batch(self, str_list):
        B = len(str_list)
        val_embs = []
        val_len = np.zeros(B, dtype=np.int64)
        for i, single_str in enumerate(str_list):
            if self.trainable:
                val = [self.w2i.get(x, 0) for x in single_str]
            else:
                val = [self.word_emb.get(x, np.zeros(
                    self.N_word, dtype=np.float32)) for x in single_str]
            val_embs.append(val)
            val_len[i] = len(val)
        max_len = max(val_len)

        if self.trainable:
            val_tok_array = np.zeros((B, max_len), dtype=np.int64)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_tok_array[i,t] = val_embs[i][t]
            val_tok = torch.from_numpy(val_tok_array)
            if self.gpu:
                val_tok = val_tok.cuda()
            val_tok_var = Variable(val_tok)
            val_inp_var = self.embedding(val_tok_var)
        else:
            val_emb_array = np.zeros(
                    (B, max_len, self.N_word), dtype=np.float32)
            for i in range(B):
                for t in range(len(val_embs[i])):
                    val_emb_array[i,t,:] = val_embs[i][t]
            val_inp = torch.from_numpy(val_emb_array)
            if self.gpu:
                val_inp = val_inp.cuda()
            val_inp_var = Variable(val_inp)

        if self.character_emb != None:
            

            col_character = []
            max_character_len = 0
            for single_list in str_list:
                single_list_res = []
                for word in single_list:
                    single_word_res = []
                    cnt = 0
                    for c in word:
                         single_word_res.append(self.c2i[c])
                         cnt += 1
                         if cnt > max_character_len:
                             max_character_len = cnt
                    single_list_res.append(single_word_res)
                col_character.append(single_list_res)


            temp = np.zeros((B, max_len, max_character_len), dtype=np.int64)
            for i in range(B):
                for j in range(len(col_character[i])):
                    for k in range(len(col_character[i][j])):
                        temp[i, j, k] = col_character[i][j][k]
            temp2 = torch.from_numpy(temp)
            if self.gpu:
                temp2 = temp2.cuda()
            temp3 = Variable(temp2)
            temp3 = temp3.view(B, -1)

            character_emb_temp = self.character_embedding(temp3)
            character_emb_temp = character_emb_temp.view(B, max_len, max_character_len, self.N_word).permute(0, 3, 1, 2)
            
            character_res1_maxperm, character_res2_maxperm, character_res3_maxperm = self.get_max_perm(character_emb_temp)


            character_res_final = torch.cat([character_res1_maxperm, character_res2_maxperm, character_res3_maxperm], -1)

            character_res_final = self.dropout(character_res_final)

            val_inp_var = torch.cat([character_res_final, val_inp_var], -1)

        return val_inp_var, val_len
