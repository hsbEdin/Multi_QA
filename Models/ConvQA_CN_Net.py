# -- coding:UTF-8 --

import math
import random
import numpy as np
import string
import re
import time
import pdb
import collections
from collections import Counter
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
from torch.nn.parameter import Parameter
from transformers import BertTokenizer, BertModel
from Models.Layers import MaxPooling, CNN, dropout, RNN_from_opt, set_dropout_prob, weighted_avg, set_seq_dropout, Attention, DeepAttention, LinearSelfAttn, GetFinalScores
from Utils.CoQAUtils import POS, ENT, tensorFromSentence
from Models.dataprocess import dataprocess, normalizeString
from rouge import *


class ConvQA_CN_Net(nn.Module):
    def __init__(self, opt, train_l, dev_l):
        super(ConvQA_CN_Net, self).__init__()
        print('ConvQA_CN_Net model\n')

        self.opt = opt
        self.use_cuda = self.opt['cuda']
        self.dropout_p = self.opt['dropout_p']
        self.hidden_size = self.opt['hidden_size']
        self.batch_size = self.opt['BATCH_SIZE']
        self.max_answer_length = self.opt['max_answer_length']
        self.train_lang = train_l
        self.dev_lang = dev_l
        self.loss_ratio = self.opt['loss_ratio']
        # set_dropout_prob(0.0 if not 'DROPOUT' in opt else float(opt['DROPOUT']))
        # set_seq_dropout('VARIATIONAL_DROPOUT' in self.opt)
        if self.opt['dataset'] == 'coqa' or self.opt['dataset'] == 'quac':
            self.pretrain_path = self.opt['english_pre_trained_dir']
            self.tokenizer = BertTokenizer.from_pretrained(self.pretrain_path)
            self.model = BertModel.from_pretrained(self.pretrain_path, hidden_dropout_prob=opt['DROPOUT'])
        else:
            self.pretrain_path = self.opt['chinese_pre_trained_dir']
            self.tokenizer = BertTokenizer.from_pretrained(self.pretrain_path)
            self.model = BertModel.from_pretrained(self.pretrain_path, hidden_dropout_prob=opt['DROPOUT'])

        self.embedding = nn.Embedding(self.opt['train_words'], self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        # vocab_dim = int(opt['VOCAB_DIM'])
        # self.pre_align = Attention(vocab_dim, opt['prealign_hidden'], correlation_func = 5, do_similarity = True)

        self.criterion = nn.NLLLoss()

        self.num_layers = 2
        self.brnn = nn.LSTM(self.hidden_size , self.hidden_size, self.num_layers, bidirectional=True)
        self.rnn = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers, bidirectional=True)
        self.Vh = nn.Parameter(torch.FloatTensor(1, self.hidden_size * 2), requires_grad = True)
        self.Wh = nn.Parameter(torch.FloatTensor(self.hidden_size * 2, self.hidden_size * 2), requires_grad = True)
        self.Uh = nn.Parameter(torch.FloatTensor(self.hidden_size * 2, self.hidden_size * 2), requires_grad = True)
        self.Vq = nn.Parameter(torch.FloatTensor(1, self.hidden_size * 2), requires_grad = True)
        self.Wq = nn.Parameter(torch.FloatTensor(self.hidden_size * 2, self.hidden_size * 2), requires_grad = True)
        self.Uq = nn.Parameter(torch.FloatTensor(self.hidden_size * 2, self.hidden_size * 2), requires_grad = True)

        # self.layernorm = nn.LayerNorm([self.batch_size, self.opt["max_featrue_length"], self.hidden_size * 2])
        self.layernorm = nn.LayerNorm([self.batch_size, self.opt["max_featrue_length"], self.hidden_size])

        self.type_out = nn.Linear(self.opt["max_featrue_length"] * 2, 2)
        self.reset_parameters()

        # self.hidden = nn.Linear(self.opt['hidden_size']*2, 2, bias=True)
        self.hidden = nn.Linear(self.opt['hidden_size'], 2, bias=True)

        self.max_length = self.opt['max_featrue_length']

        self.generate_embedding = nn.Embedding(self.opt['dev_words'], self.hidden_size * 2)
        self.attn = nn.Linear(self.hidden_size * 4, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 4, self.hidden_size * 2)
        self.gru = nn.GRU(self.hidden_size * 2, self.hidden_size * 2)
        self.out = nn.Linear(self.hidden_size * 2, self.opt['dev_words'])


    def reset_parameters(self):
        nn.init.xavier_uniform_(self.Vh, gain=1)
        nn.init.xavier_uniform_(self.Vq, gain=1)
        nn.init.xavier_uniform_(self.Wh, gain=1)
        nn.init.xavier_uniform_(self.Wq, gain=1)
        nn.init.xavier_uniform_(self.Uh, gain=1)
        nn.init.xavier_uniform_(self.Uq, gain=1)
        nn.init.xavier_uniform_(self.type_out.weight, gain=1)

    def forward(self, x, x_bert_mask, q, q_mask, his_info_list, answer_strs, ground_truth, context_str, context_ids, turn_ids, answer_types, is_max_context, is_training=False):
        x_bert = torch.cat(x, 0)
        x_mask = torch.cat(x_bert_mask, 0)
        # print(self.model(x_bert)[0])
        x_bert = self.model(x_bert, attention_mask=x_mask)[0]

        new_his, new_his_mask = [], []
        his_sign = []
        for his in his_info_list:
            new_his += his['his']
            new_his_mask += his['his_mask']
            if his['his'][0][0][0].data == 0:
                his_sign.append(0)
            else:
                his_sign.append(1)

        ### 用his[0]==0 找出0
        history = torch.cat(new_his, 0)
        history_mask = torch.cat(new_his_mask, 0)
        q_tensor = torch.cat(q, 0)
        q_mask = torch.cat(q_mask, 0)

        history_length = history.size()[1]
        q_length = q_tensor.size()[1]

        history = self.embedding(history).view(history_length, self.batch_size, -1)
        q_tensor = self.embedding(q_tensor).view(q_length, self.batch_size, -1)

        history, _ = self.rnn(history)
        q_tensor, _ = self.rnn(q_tensor)

        history = self.model(history, attention_mask=history_mask)[0]
        q_tensor = self.model(q_tensor, attention_mask=q_mask)[0]
        # history = history.repeat([1,1,2]).cuda()
        # q_tensor = q_tensor.repeat([1, 1, 2]).cuda()
        # x_list = torch.unbind(x_bert, 0)
        # x_final_list = []
        # for x, his in zip(x_list, his_info_list):
        #     if his:
        #         his["his"] = torch.cat(his["his"], 0)
        #         his["his"] = self.model(his["his"])[0]
        #         his["his_mask"] = torch.cat(his["his_mask"], 0)
        #         his_size = his["his"].size()[0]
        #         x_bert_ex = x.expand(his_size, -1, -1)
        #         # print(x_bert_ex.size(), his["his"].size())
        #         # exit(0)
        #         x_align = self.pre_align(x_bert_ex, his["his"], his["his_mask"])
        #         x_final = torch.mean(x_align, dim=0)
        #         x_final = x_final.unsqueeze(0)  # 1, 384,768
        #         x_final = x + F.relu(x_final)
        #         x_final_list.append(x_final)
        #     else:
        #         x = torch.unsqueeze(x, 0)
        #         x_final_list.append(x)
        # x_final = torch.cat(x_final_list, 0)
        # x_final = self.layernorm(x_final).cuda()


        x_bert = x_bert.view(-1, self.batch_size, self.hidden_size)
        x_bert, _ = self.brnn(x_bert)
        history = history.view(-1, self.batch_size, self.hidden_size)
        bi_history, _ = self.brnn(history)
        q_tensor = q_tensor.view(-1, self.batch_size, self.hidden_size)
        bi_q_tensor, _ = self.brnn(q_tensor)

        bi_history = bi_history.view(self.batch_size, -1, self.hidden_size * 2)
        bi_q_tensor = bi_q_tensor.view(self.batch_size, -1, self.hidden_size * 2)
        # print("q_tensor: ", q_tensor)
        ### Attention
        self.Vh.data = self.Vh.squeeze(0)
        self.Vq.data = self.Vq.squeeze(0)
        # x_h = self.pre_align(x_bert.view(self.batch_size, -1, self.hidden_size * 2), history, history_mask)
        xh = self.attention(x_bert, x_mask, bi_history, history_mask, self.Vq, self.Wq, self.Uq)
        x_h = []
        for i, s in enumerate(his_sign):
            tmp = x_bert.view(self.batch_size, -1, self.hidden_size * 2)[i]
            if s==0:
                x_h.append(tmp.unsqueeze(0))
            else:
                x_h.append((tmp + F.relu(xh[i])).unsqueeze(0))
        x_h = torch.cat(x_h, 0)
        x_h_q = self.attention(x_h.view(-1, self.batch_size, self.hidden_size * 2), x_mask, bi_q_tensor, q_mask, self.Vq, self.Wq, self.Uq)
        x_h_q = x_bert.view(self.batch_size, -1, self.hidden_size * 2) + F.relu(x_h_q)

        x_h = self.layernorm(x_h).type(torch.FloatTensor).cuda()
        x_h_q = self.layernorm(x_h_q).type(torch.FloatTensor).cuda()

        x_h = self.dropout(x_h)
        x_h_q = self.dropout(x_h_q)
        # value0 = torch.mean(torch.mean(x_h, 2), 1)
        # value1 = torch.mean(torch.mean(x_h_q, 2), 1)
        # #
        extract = []
        generate = []
        type_loss = []
        for i, (v1, v2, t) in enumerate(zip(x_h, x_h_q, answer_types)):
            if t == 'extractive':
                ans = torch.tensor([1, 0]).float().cuda()
            else:
                ans = torch.tensor([0, 1]).float().cuda()

            o1 = F.avg_pool1d(v1.unsqueeze(0), kernel_size=self.hidden_size * 2).squeeze(2)
            o2 = F.avg_pool1d(v2.unsqueeze(0), kernel_size=self.hidden_size * 2).squeeze(2)
            output = self.type_out(torch.cat((o1, o2), 1)).squeeze(0)
            tmp_loss = F.binary_cross_entropy(torch.sigmoid(output), ans)
            type_loss.append(tmp_loss)
            x, y = output.tolist()
            if x>y:
                extract.append((i, x_h[i].unsqueeze(0)))
            else:
                generate.append((i, x_h_q[i].unsqueeze(0)))
        for i in range(self.batch_size):
            extract.append((i, x_final[i].unsqueeze(0)))

        extractive, generative = [], []
        ex_id, ge_id = collections.OrderedDict(), collections.OrderedDict()

        for i, (id, ex) in enumerate(extract):
            extractive.append(ex)
            ex_id[i] = id
        for i, (id, ge) in enumerate(generate):
            generative.append(ge)
            ge_id[i] = id

        if not is_training:
            if not generative:
                ex_tensor = torch.cat(extractive, 0)
                loss_list, f1_list, pred_json, do_pre = self.extract_evaluate(ex_tensor, ex_id, ground_truth, answer_strs, context_str, context_ids, turn_ids, type_loss, is_max_context)
                if loss_list:
                    loss = torch.cat(loss_list, 0)
                    loss = torch.mean(loss, dim=0)
                    loss = round(loss.item(), 4)
                    loss_list = loss
                return loss_list, f1_list, pred_json, do_pre
            elif not extractive:
                ge_tensor = torch.cat(generative, 0)
                loss_list, f1_list, pred_json = self.generate_evaluate(ge_tensor, ge_id, answer_strs, context_ids, turn_ids, type_loss)
                if not loss_list:
                    return [], [], [], False
                loss = torch.cat(loss_list, 0)
                loss = torch.mean(loss, dim=0)
                loss = round(loss.item(), 4)
                avg_f1 = np.average(f1_list)
                while len(f1_list) < self.batch_size:
                    f1_list.append(avg_f1)
                return loss, f1_list, pred_json, True
            else:
                ex_tensor = torch.cat(extractive, 0)
                ge_tensor = torch.cat(generative, 0)
                ex_loss_list, ex_f1_list, ex_pred_json, do_pre = self.extract_evaluate(ex_tensor, ex_id, ground_truth, answer_strs, context_str, context_ids, turn_ids, type_loss, is_max_context)
                ge_loss_list, ge_f1_list, ge_pred_json = self.generate_evaluate(ge_tensor, ge_id, answer_strs, context_ids, turn_ids, type_loss)
                if not ex_loss_list:
                    loss = torch.cat(ge_loss_list, 0)
                    loss = torch.mean(loss, dim=0)
                    loss = round(loss.item(), 4)

                    avg_f1 = np.average(ge_f1_list)
                    while len(ge_f1_list) < self.batch_size:
                        ge_f1_list.append(avg_f1)
                    f1_list = ge_f1_list
                    return loss, f1_list, ge_pred_json, True
                elif not ge_loss_list:
                    loss = torch.cat(ex_loss_list, 0)
                    loss = torch.mean(loss, dim=0)
                    loss = round(loss.item(), 4)

                    avg_f1 = np.average(ex_f1_list)
                    while len(ex_loss_list) < self.batch_size:
                        ex_loss_list.append(avg_f1)
                    f1_list = ex_loss_list
                    return loss, f1_list, ex_pred_json, True
                else:
                    loss_list = ex_loss_list + ge_loss_list
                    loss = torch.cat(loss_list, 0)
                    loss = torch.mean(loss, dim=0)
                    loss = round(loss.item(), 4)
                    f1_list = ex_f1_list + ge_f1_list
                    avg_f1 = np.average(f1_list)
                    while len(f1_list) < self.batch_size:
                        f1_list.append(avg_f1)
                    pred_json = ex_pred_json + ge_pred_json
                    return loss, f1_list, pred_json, True


        if not generative:
            ex_tensor = torch.cat(extractive, 0)
            ex_loss = self.extract_loss(ex_tensor, ex_id, ground_truth, type_loss)
            return ex_loss
        elif not extractive:
            ge_tensor = torch.cat(generative, 0)
            ge_loss = self.trainIters(ge_tensor, ge_id, answer_strs, type_loss)
            return ge_loss
        else:
            ex_tensor = torch.cat(extractive, 0)
            ge_tensor = torch.cat(generative, 0)
            ex_loss = self.extract_loss(ex_tensor, ex_id, ground_truth, type_loss)
            ge_loss = self.trainIters(ge_tensor, ge_id, answer_strs, type_loss)
            for i, (id, ge) in enumerate(generate):
                ex_loss.insert(id, ge_loss[i])

            return ex_loss
        # ex_id = [i for i in range(len(x_h))]
        # ex_loss = self.extract_loss(x_h, ex_id, ground_truth)
        # return ex_loss


    def attention(self, x_tensor, x_mask, y_tensor, y_mask, V, W, U):
        '''
        :param x_tensor: x_length * 16 * 1536
        :param x_mask: 16 * 384
        :param y_tensor: 16 * y_length * 1536
        :param y_mask: 16 * y_length
        :param V: 1 * 1536
        :param W: 1536 * 1536
        :param U: 1536 * 1536
        :return: 16 * x_length * 1536
        '''
        x_list = []
        y = y_tensor

        for i, x in enumerate(x_tensor):
            x = x.unsqueeze(0).contiguous().view(self.batch_size, -1, self.hidden_size * 2).cuda()
            s = torch.sum(V * torch.tanh(torch.matmul(x, W) + torch.matmul(y_tensor, U)), 2)

            s = s.unsqueeze(0)
            x_list.append(s)
        matrix = torch.cat(x_list, 0).permute(1,0,2) # 16 * x_tensor length * y_tensor length
        # xmask = x_mask.unsqueeze(2).float().cuda()
        # ymask = y_mask.unsqueeze(1).float().cuda()
        # empty_mask = torch.matmul(xmask, ymask).long().eq(0)

        # print(a.size(), a)
        empty_mask = y_mask.eq(0).unsqueeze(1).expand_as(matrix).cuda()
        # print('empty_mask: ', empty_mask.size(), empty_mask)



        matrix.data.masked_fill_(empty_mask.data, -float('inf'))
        # print("1:", F.softmax(matrix, dim=1))
        # print("2:", F.softmax(matrix, dim=2))
        # print("matrix: ", matrix)
        alpha_flat = F.softmax(matrix.reshape(-1, y_tensor.size(1)), dim=1) ### NAN
        # print("alpha_flat: ", alpha_flat)
        alpha_flat = torch.where(torch.isnan(alpha_flat), torch.full_like(alpha_flat, 0), alpha_flat)
        alpha = alpha_flat.reshape(-1, x_tensor.size(0), y_tensor.size(1))
        score = alpha.bmm(y)
        return score

    def extract_loss(self, extractive, ex_id, ground_truth, type_loss):
        batch_size = extractive.size()[0]
        seq_length = extractive.size()[1]
        hidden_size = extractive.size()[2]

        final_hidden_matrix = torch.reshape(extractive, [batch_size * seq_length, hidden_size])
        logits = self.hidden(final_hidden_matrix)
        logits = torch.reshape(logits, [batch_size, seq_length, 2])

        logits = logits.permute([2, 0, 1])

        unstacked_logits = torch.unbind(logits, 0)

        (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

        # start_logits, end_logits
        batch_size = start_logits.size()[0]
        seq_length = start_logits.size()[1]
        loss_list = []

        for i in range(batch_size):
            start_positions = ground_truth[ex_id[i]][0]
            end_positions = ground_truth[ex_id[i]][1]
            s_prob = F.softmax(start_logits[i], dim=-1)
            e_prob = F.softmax(end_logits[i], dim=-1)
            s, e = s_prob.tolist(), e_prob.tolist()
            sp, ep = s.index(max(s)), e.index(max(e))

            start_loss, s_prob = self.compute_loss(start_logits[i], seq_length, start_positions)
            end_loss, e_prob = self.compute_loss(end_logits[i], seq_length, end_positions)

            # ans = answer_strs[i]
            # if sp<ep:
            #     if self.opt['dataset'] == 'coqa' or self.opt['dataset'] == 'quac':
            #         # print("ans: ", "".join(ans))
            #         # print("pre: ", self.process_ans("".join(context_str[i][sp:ep])))
            #         if type(answer_strs[i]) == str:
            #             f1 = evaluate.compute_f1(ans, self.process_ans(" ".join(context_str[i][sp:ep])))
            #         else:
            #             f1 = evaluate.compute_f1("".join(ans), self.process_ans(" ".join(context_str[i][sp:ep])))
            #         # print("f1: ", f1)
            #         # print("\n")
            #     # else:
            #     #     print("ans: ", "".join(ans))
            #     #     print("pre: ", self.process_ans("".join(context_str[i])[sp:ep]))

            # loss = (1 - self.loss_ratio) * ((start_loss + end_loss) / 2.0) + self.loss_ratio * type_loss[ex_id[i]]
            loss = (start_loss + end_loss) / 2.0

            # if sp<ep:
            #     print("ans: ", context_words[i][start_positions:end_positions])
            #     print("pre: ", sp, ep, context_words[i][sp:ep])

            loss_list.append(loss.unsqueeze(0))
        return loss_list

    def compute_loss(self, logits, seq_length, positions):
        pos = torch.LongTensor(1, 1)
        # if positions == None:
        #     one_hot_positions = torch.zeros(1, seq_length, dtype=torch.float32)
        # else:
        pos[0, 0] = positions
        one_hot_positions = torch.zeros(1, seq_length, dtype=torch.float32).scatter_(1, pos, 1)

        if self.opt['cuda']:
            one_hot_positions = Variable(one_hot_positions.cuda())
        log_probs = F.log_softmax(logits, dim=-1)
        loss = -torch.mean(torch.sum(one_hot_positions * log_probs, dim=-1))
        return loss, log_probs

    def generate_decoder(self, input, hidden, encoder_outputs):

        embedded = self.generate_embedding(input).view(1, 1, -1)
        # embedded = self.dropout(embedded)
        hidden = hidden.view(1, 1, -1)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden_state = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden_state, attn_weights

    def generative_loss(self, batch_size, target_tensor, encoder_output, ge_id, type_loss):
        #
        # seq_len = encoder_output.size(1)
        # hidden = encoder_output.view(seq_len, batch_size, -1)
        # decoder_hidden = hidden[0].view(batch_size, 1, -1)
        decoder_hidden = torch.zeros(batch_size, 1, self.hidden_size * 2).cuda()

        # decoder_optimizer.zero_grad()

        total_loss = []

        ### encoder_output:
        #   seq_length * hidden_size
        #   384 * 768

        decoder_input = torch.zeros([batch_size, 1, 1], dtype=torch.long).cuda()  # 16 * 0

        ### target_tensor是一个List, 里面是batch数量的tensor
        for i in range(batch_size):
            hidden = decoder_hidden[i].cuda()
            input = decoder_input[i].cuda()
            loss = 0
            for di in range(len(target_tensor[i])):
                target = target_tensor[i][di].cuda()
                # print(decoder_input[i].size(), decoder_hidden[i].size(), encoder_output[i].size(), target[di].size())
                # exit(0)
                decoder_output, hidden, decoder_attention = self.generate_decoder(
                    input, hidden, encoder_output[i])
                loss = loss + self.criterion(decoder_output, target)
                input = target

            # loss = (1 - self.loss_ratio) * (loss / len(target)) + self.loss_ratio * type_loss[ge_id[i]]
            loss = (loss / len(target))

            total_loss.append(loss)
        return total_loss

    def trainIters(self, encoder_output, ge_id, answer_strs, type_loss):
        target = []

        for k, v in ge_id.items():
            answers = normalizeString(answer_strs[v])
            tmp, _ = tensorFromSentence(self.dev_lang, answers, len(answers))
            tmp = tmp.view(-1, 1)
            target.append(tmp)
        batch_size = encoder_output.size()[0]


        # print("batch_size: ", batch_size)
        # print("target_tensor: ", target_tensor.size())
        # print("encoder_output: ", encoder_output.size())

        loss = self.generative_loss(batch_size, target, encoder_output, ge_id, type_loss)
        return loss

    def extract_evaluate(self, extractive, ex_id, ground_truth, answer_strs, context_str, context_ids, turn_ids, type_loss, is_max_context):
        # evaluate.py 用来做中文
        # if self.opt['dataset'] == 'coqa':
        #     gold_file = self.opt['CoQA_DEV_FILE']
        #     evaluate = CoQAEvaluator(self.opt, gold_file)
        # elif self.opt['dataset'] == 'chinese':
        #     gold_file = self.opt['DEV_FILE']
        #     evaluate = CoQAEvaluator(self.opt, gold_file)

        batch_size = extractive.size()[0]
        seq_length = extractive.size()[1]
        hidden_size = extractive.size()[2]
        list_len = batch_size

        final_hidden_matrix = torch.reshape(extractive, [batch_size * seq_length, hidden_size])
        logits = self.hidden(final_hidden_matrix)
        logits = torch.reshape(logits, [batch_size, seq_length, 2])

        logits = logits.permute([2, 0, 1])

        unstacked_logits = torch.unbind(logits, 0)

        (start_logits, end_logits) = (unstacked_logits[0], unstacked_logits[1])

        _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "PrelimPrediction",
            ["start_index", "end_index", "start_logit", "end_logit"])

        n_best_size = self.opt['N_BEST']
        loss_list = []
        f1_list = []
        pred_json = []
        do_pre = True  # 用于判断是否有feature
        prelim_predictions = []

        for i in range(batch_size):
            ### start, end 都是None
            start_positions = ground_truth[ex_id[i]][0]
            end_positions = ground_truth[ex_id[i]][1]

            if start_positions == None:
                continue
                ### ans = "无法回答"
                ### 有可能会跳过所有判断
            if start_positions >= seq_length or end_positions >= seq_length:
                continue

            start_loss, _ = self.compute_loss(start_logits[i], seq_length, start_positions)
            end_loss, _ = self.compute_loss(end_logits[i], seq_length, end_positions)

            s_prob = F.softmax(start_logits[i], dim=-1)
            e_prob = F.softmax(end_logits[i], dim=-1)
            s, e = s_prob.tolist(), e_prob.tolist()
            sp, ep = s.index(max(s)), e.index(max(e))
            start_indexes = self._get_best_indexes(start_logits[i], n_best_size)
            end_indexes = self._get_best_indexes(end_logits[i], n_best_size)
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= seq_length:
                        continue
                    if end_index >= seq_length:
                        continue
                    if end_index < start_index:
                        continue
                    if not is_max_context[i].get(start_index, False):
                        continue
                    length = end_index - start_index + 1
                    if length > self.opt['ans_max_len']:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=start_logits[i][start_index],
                            end_logit=end_logits[i][end_index],
                        ))
            if prelim_predictions:
                prelim_predictions = sorted(
                    prelim_predictions,
                    key=lambda x: (x.start_logit + x.end_logit),
                    reverse=True)
                _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
                    "NbestPrediction", ["text", "start_logit", "end_logit"])
                nbest = []
                pred = prelim_predictions[0]
                if len(nbest) >= n_best_size:
                    continue
                if self.opt['dataset'] == 'chinese':
                    context = "".join(context_str[i])
                    tok_tokens = context[pred.start_index:(pred.end_index + 1)]
                    tok_text = tok_tokens
                else:
                    tok_tokens = context_str[i][pred.start_index:(pred.end_index + 1)]
                    tok_text = " ".join(tok_tokens)

                orig_tokens = answer_strs[ex_id[i]]

                if self.opt['dataset'] == ("coqa" or 'quac'):
                    tok_text = tok_text.replace(" ##", "")
                    tok_text = tok_text.replace("##", "")
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = orig_tokens

                if self.opt['dataset'] == 'quac' or self.opt['dataset'] == 'coqa':
                    f1 = self.f1_score(tok_text, orig_text)
                else:
                    f1 = evaluate.compute_f1(orig_text, tok_text, self.opt['dataset'])
                    # f1 = self.f1_score(tok_text, orig_text)
                # if f1>0:
                #     print("tok_text: ", tok_text)
                #     print("orig_text: ", orig_text)
                #     print("f1: ", f1)
                # else:
                #     f1 = evaluate.compute_f1(ans, context_str[i][sp:ep])
                # loss = (1 - self.loss_ratio) * ((start_loss + end_loss) / 2.0) + self.loss_ratio * type_loss[ex_id[i]]
                loss = (start_loss + end_loss) / 2.0

                loss_list.append(loss.unsqueeze(0))
                f1_list.append(f1)
                pred_json.append({
                    'id': context_ids[i],
                    'turn_id': turn_ids[i],
                    'answer': orig_text,
                    'predict': tok_text,
                    'type': 'extractive'
                })

        if loss_list:

            ### 最后一轮的batch_size较小，给最后一轮补齐
            avg_f1 = np.average((f1_list))
            while len(f1_list) < list_len:
                f1_list.append(avg_f1)

            return loss_list, f1_list, pred_json, do_pre
        else:
            do_pre = False
            return [], [], [], do_pre

    def generate_evaluate(self, encoder_output, ge_id, answer_strs, context_ids, turn_ids, type_loss):
        # with torch.no_grad():
        target_tensor = []
        EOS_token = 1
        SOS_token = 0
        max_length = self.max_answer_length
        answer_words = []
        for k, v in ge_id.items():
            answers = normalizeString(answer_strs[v])
            answer_words.append(answers)
            tmp, _ = tensorFromSentence(self.dev_lang, answers, len(answers))
            tmp = tmp.view(-1, 1)
            target_tensor.append(tmp)

        batch_size = encoder_output.size(0)

        # decoder_input = torch.ones([batch_size, 1], dtype=torch.long).cuda()# SOS 1, 1

        decoder_hidden = torch.zeros(batch_size, 1, self.hidden_size * 2).cuda()

        words_list = []
        loss_list = []
        pred_json = []
        f1_list = []
        rouge_list = []
        prelim_predictions = []
        # decoder_attentions = torch.zeros(max_length, max_length)
        for i in range(batch_size):
            decoded_words = []
            hidden = decoder_hidden[i].cuda()
            decoder_input = torch.tensor([[SOS_token]], dtype=torch.long).cuda()
            loss = 0
            for di in range(max_length):

                decoder_output, hidden, decoder_attention = self.generate_decoder(
                    decoder_input, hidden, encoder_output[i])
                # decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(self.dev_lang.index2word[topi.item()])
                if len(target_tensor[i])<=di:
                    break
                target = target_tensor[i][di].cuda()
                loss = loss + self.criterion(decoder_output, target)
                decoder_input = topi.squeeze().detach()
            loss /= len(target)
            if type(loss) == float:
                words_list.append('')
                continue
            rouge_result = rouge_max_over_ground_truths(" ".join(decoded_words), answer_strs[ge_id[i]])
            words_list.append(" ".join(decoded_words))
            # loss_list.append((1-self.loss_ratio) * loss.unsqueeze(0) + self.loss_ratio * type_loss[ge_id[i]])
            loss_list.append(loss.unsqueeze(0))

            pred_json.append({
                'id': context_ids[i],
                'turn_id': turn_ids[i],
                'answer': answer_words[i],
                'predict': " ".join(decoded_words),
                'type: ': "generative",
                'rouge_result': rouge_result
            })

        for x, y in zip(words_list, answer_words):
            if x == '': continue
            f1 = self.f1_score(x, y)
            f1_list.append(f1)
        return loss_list, f1_list, pred_json

    def normalize_answer(self, s):
        """Lower text and remove punctuation, articles and extra whitespace."""

        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    def f1_score(self, prediction, ground_truth):
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def _get_best_indexes(self, logits, n_best_size):
        """Get the n-best logits from a list."""
        index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

        best_indexes = []
        for i in range(len(index_and_score)):
            if i >= n_best_size:
                break
            best_indexes.append(index_and_score[i][0])
        return best_indexes

    def rouge_max_over_ground_truths(self, prediction, ground_truths):
        scores_for_rouge1 = []
        scores_for_rouge2 = []
        scores_for_rougel = []
        for ground_truth in ground_truths:
            score = cal_rouge([prediction], [ground_truth])
            scores_for_rouge1.append(score[0])
            scores_for_rouge2.append(score[1])
            scores_for_rougel.append(score[2])
        return max(scores_for_rouge1), max(scores_for_rouge2), max(scores_for_rougel)

    def cal_rouge(self, infer, ref):
        x = rouge.rouge(infer, ref)
        return x['rouge_1/f_score'] * 100, x['rouge_2/f_score'] * 100, x['rouge_l/f_score'] * 100

    ### LSTM decoder 三维矩阵代码
    # def generate_decoder(self, input, hidden, encoder_outputs):
    #     batch_size = encoder_outputs.size(0)
    #     """
    #         input: 16 * 1
    #         hidden: 16 * 1 * 768
    #         encoder_outputs: 16 * 384 * 768
    #     """
    #     embedded = self.generate_embedding(input).view(batch_size, 1, -1)
    #     embedded = self.dropout(embedded)
    #
    #     attn_weights = F.softmax(
    #         self.attn(torch.cat((embedded, hidden), 2)), dim=2)
    #     attn_applied = torch.bmm(attn_weights, encoder_outputs)
    #
    #     output = torch.cat((embedded, attn_applied), 2)
    #     output = self.attn_combine(output)
    #     output = F.relu(output)
    #
    #     output, hidden = self.gru(output, hidden)
    #
    #     output = F.log_softmax(self.out(output), dim=2)
    #     """
    #             output: [16, 1, 768]
    #             hidden: [16, 1, 768]
    #             attn_weights: [16, 1, 384]
    #             """
    #
    #     return output, hidden, attn_weights