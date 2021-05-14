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
from torch.nn import CrossEntropyLoss
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as init
from torch.nn.parameter import Parameter
from transformers import BertTokenizer, BertForQuestionAnswering, BertModel
from Models.Layers import MaxPooling, CNN, dropout, RNN_from_opt, set_dropout_prob, weighted_avg, set_seq_dropout, Attention, DeepAttention, LinearSelfAttn, GetFinalScores
from Utils.CoQAUtils import POS, ENT, tensorFromSentence
from Models.dataprocess import dataprocess, normalizeString
from rouge import *
from model.modeling_auto import MODEL_FOR_CONVERSATIONAL_QUESTION_ANSWERING_MAPPING, AutoModelForConversationalQuestionAnswering



class ConvQA_CN_Net(nn.Module):
    def __init__(self, opt):
        super(ConvQA_CN_Net, self).__init__()
        print('ConvQA_CN_Net model\n')
        self.opt = opt

        if self.opt['dataset'] == 'coqa' or self.opt['dataset'] == 'quac':
            self.pretrain_path = self.opt['base_pre_trained_dir']
            # self.model = BertModel.from_pretrained(self.pretrain_path)
            # self.pretrain_path = self.opt['finetuned_pre_trained_dir']
            # self.model = BertForQuestionAnswering.from_pretrained(self.pretrain_path)
            self.model = AutoModelForConversationalQuestionAnswering.from_pretrained(
                self.pretrain_path,
                from_tf=bool(".ckpt" in self.pretrain_path),
                config=self.pretrain_path,
                cache_dir=None,
            )

        # if 'LOCK_BERT' in self.opt:
        #     print('Lock BERT\'s weights')
        #     for name, p in self.model.named_parameters():
        #         if name != "qa_outputs":
        #             p.requires_grad = False
        # new = []
        # for name, p in self.model.named_parameters():
        #     new.append((name, p, p.requires_grad))
        # pdb.set_trace()
        self.dropout_p = self.opt['dropout_p']
        self.hidden_size = self.opt['hidden_size']
        self.batch_size = self.opt['BATCH_SIZE']
        self.max_answer_length = self.opt['max_answer_length']
        self.loss_ratio = self.opt['loss_ratio']

        self.num_layers = 2
        # self.brnn = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers)
        # self.qa_outputs = nn.Linear(self.hidden_size, 2)
        # self.model.init_weights()
        # self.layernorm = nn.LayerNorm([self.batch_size, self.opt["max_featrue_length"], self.hidden_size])

    def forward(self, x, x_bert_mask, x_segment, q, q_mask, his_info_list, answer_strs, ex_pre, ground_truth, context_str, context_ids, turn_ids, answer_types, is_max_context, token_to_orig_map, is_training=False):

        x_bert = torch.cat(x, 0)
        x_mask = torch.cat(x_bert_mask, 0)
        x_sep = torch.cat(x_segment, 0)
        ### 普通bert
        # print(self.model(x_bert)[-1])
        # x_bert = self.model(x_bert, attention_mask=x_mask, token_type_ids=x_sep)[0]
        ### qa_bert
        x_bert = self.model(x_bert, attention_mask=x_mask, token_type_ids=x_sep, output_hidden_states=True)
        x_bert = x_bert.hidden_states[-1]
        ### qa conversation bert
        
        # x_bert = x_bert.view(-1, self.batch_size, self.hidden_size)
        # x_rnn, _ = self.brnn(x_bert)
        # x_bert = (x_bert + F.relu(x_rnn)).view(self.batch_size, -1, self.hidden_size)
        # x_bert = self.layernorm(x_bert)

        if not is_training:
            loss_list, f1_list, pred_json, do_pre = self.extract_evaluate(x_bert, ground_truth, ex_pre, context_str, context_ids, turn_ids, is_max_context, token_to_orig_map)
            if loss_list:
                loss = torch.cat(loss_list, 0)
                loss = torch.mean(loss, dim=0)
                loss = round(loss.item(), 4)
                loss_list = loss
            return loss_list, f1_list, pred_json, do_pre


        else:
            ex_loss = self.extract_loss(x_bert, ground_truth, context_str)
            return ex_loss

    def extract_loss(self, extractive, ground_truth, context_str):

        loss_list = []
        for i in range(self.batch_size):
            logits = self.model.qa_outputs(extractive[i].unsqueeze(0))
            # logits = self.qa_outputs(extractive[i].unsqueeze(0))
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)
            start_positions = ground_truth[i][0]
            end_positions = ground_truth[i][1]

            pre = ' '.join(context_str[i][torch.argmax(start_logits): torch.argmax(end_logits) + 1])
            real = ' '.join(context_str[i][start_positions: end_positions])
            # print("predict: ", pre)
            # print("answer: ", real)
            # print("\n")
            start_positions = start_positions.unsqueeze(0).cuda()
            end_positions = end_positions.unsqueeze(0).cuda()
            # pdb.set_trace()
            if start_positions is not None and end_positions is not None:
                # If we are on multi-GPU, split add a dimension
                if len(start_positions.size()) > 1:
                    start_positions = start_positions.squeeze(-1)
                if len(end_positions.size()) > 1:
                    end_positions = end_positions.squeeze(-1)
                # sometimes the start/end positions are outside our model inputs, we ignore these terms
                ignored_index = start_logits.size(1)
                start_positions.clamp_(0, ignored_index)
                end_positions.clamp_(0, ignored_index)

                loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
                start_loss = loss_fct(start_logits, start_positions)
                end_loss = loss_fct(end_logits, end_positions)

                loss = (start_loss + end_loss) / 2.0

                loss_list.append(loss.unsqueeze(0))
                # s, e = torch.argmax(start_logits), torch.argmax(end_logits)+1
                # answer = context[i][s: e]
                # print(answer)
                # print(context[i][ground_truth[i][0]: ground_truth[i][1]])
                # loss = (1 - self.loss_ratio) * ((start_loss + end_loss) / 2.0) + self.loss_ratio * type_loss[ex_id[i]]

            # if sp<ep:
            #     print("ans: ", context_words[i][start_positions:end_positions])
            #     print("pre: ", sp, ep, context_words[i][sp:ep])

        pdb.set_trace()
        return loss_list

    # def compute_loss(self, logits, seq_length, positions):
    #     pos = torch.LongTensor(1, 1)
    #     # if positions == None:
    #     #     one_hot_positions = torch.zeros(1, seq_length, dtype=torch.float32)
    #     # else:
    #     pos[0, 0] = positions
    #     one_hot_positions = torch.zeros(1, seq_length, dtype=torch.float32).scatter_(1, pos, 1).squeeze(0)
    #
    #     if self.opt['cuda']:
    #         one_hot_positions = Variable(one_hot_positions.cuda())
    #     log_probs = F.log_softmax(logits, dim=-1)
    #     loss = -torch.mean(torch.sum(one_hot_positions * log_probs, dim=-1))
    #
    #     return loss, log_probs

    def extract_evaluate(self, extractive, ground_truth, answer_strs, context_str, context_ids, turn_ids, is_max_context, token_to_orig_map):

        _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "PrelimPrediction",
            ["start_index", "end_index", "start_logit", "end_logit"])
        max_length = extractive.size(1)
        n_best_size = self.opt['N_BEST']
        loss_list = []
        f1_list = []
        pred_json = []
        do_pre = True  # 用于判断是否有feature
        prelim_predictions = []

        for i in range(self.batch_size):
            logits = self.model.qa_outputs(extractive[i].unsqueeze(0))
            # logits = self.qa_outputs(extractive[i].unsqueeze(0))
            start_logits, end_logits = logits.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1) # size: 1 * 384
            end_logits = end_logits.squeeze(-1)
            ### start, end 都是None
            start_positions = ground_truth[i][0]
            end_positions = ground_truth[i][1]
            if start_positions == None or end_positions == None:
                continue
            start_positions = torch.LongTensor([start_positions])
            end_positions = torch.LongTensor([end_positions])
            start_positions = start_positions.unsqueeze(0).cuda()
            end_positions = end_positions.unsqueeze(0).cuda()

            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            # predictions = []
            # expand_score = self.gen_upper_triangle(start_logits, end_logits, max_length)
            # prob = F.softmax(expand_score, dim=1)
            # _, ids = torch.sort(prob[0, :], descending=True)
            # idx = 0
            # best_id = ids[idx]
            # context_len = len(context_str[i])
            #
            # if best_id == context_len * context_len:
            #     predictions.append('no')
            #
            # if best_id == context_len * context_len + 1:
            #     predictions.append('yes')
            #
            # if best_id == context_len * context_len + 2:
            #     predictions.append('unknown')

            start_logits = start_logits.squeeze(0)
            end_logits = end_logits.squeeze(0)
            start_indexes = self._get_best_indexes(start_logits, n_best_size)
            end_indexes = self._get_best_indexes(end_logits, n_best_size)

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(context_str[i]):
                        continue
                    if end_index >= len(context_str[i]):
                        continue
                    if start_index not in token_to_orig_map[i]:
                        continue
                    if end_index not in token_to_orig_map[i]:
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
                            start_logit=start_logits[start_index],
                            end_logit=end_logits[end_index],
                        ))

            # if predictions:
            #     tok_text = predictions[0]
            #     orig_tokens = answer_strs[i]
            #     orig_text = " ".join(orig_tokens)
            #     f1 = self.f1_score(tok_text, orig_text)
            #     loss = (start_loss + end_loss) / 2.0
            #     loss_list.append(loss.unsqueeze(0))
            #     f1_list.append(f1)
            #     pred_json.append({
            #         'id': context_ids[i],
            #         'turn_id': turn_ids[i],
            #         'answer': orig_text,
            #         'predict': tok_text,
            #         'type': 'extractive'
            #     })

            if prelim_predictions:
                prelim_predictions = sorted(
                    prelim_predictions,
                    key=lambda x: (x.start_logit + x.end_logit),
                    reverse=True)
                _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
                    "NbestPrediction", ["text", "start_logit", "end_logit"])
                start, end = prelim_predictions[0].start_index, prelim_predictions[0].end_index
                tok_tokens = context_str[i][start: end]
                orig_tokens = answer_strs[i]
                tok_tokens = self.remove_bert_token(tok_tokens)
                orig_tokens = self.remove_bert_token(orig_tokens)
                tok_text = " ".join(tok_tokens)
                orig_text = " ".join(orig_tokens)

                if self.opt['dataset'] == 'quac' or self.opt['dataset'] == 'coqa':
                    f1 = self.f1_score(tok_text, orig_text)
                else:
                    f1 = evaluate.compute_f1(orig_text, tok_text, self.opt['dataset'])
                # print("predict: ", tok_text)
                # print("answer: ", orig_text)
                # print("f1: ", f1)
                # print("\n")
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
        # pdb.set_trace()
        if loss_list:

            # ### 最后一轮的batch_size较小，给最后一轮补齐
            avg_f1 = np.average((f1_list))
            while len(f1_list) < self.batch_size:
                f1_list.append(avg_f1)

            return loss_list, f1_list, pred_json, do_pre
        else:
            do_pre = False
            return [], [], [], do_pre

    def gen_upper_triangle(self, score_s, score_e, max_len):
        batch_size = score_s.shape[0]
        context_len = score_s.shape[1]
        # batch x context_len x context_len
        expand_score = score_s.unsqueeze(2).expand([batch_size, context_len, context_len]) + \
                       score_e.unsqueeze(1).expand([batch_size, context_len, context_len])
        score_mask = torch.ones(context_len)
        score_mask = score_mask.cuda()
        score_mask = torch.ger(score_mask, score_mask).triu().tril(max_len - 1)
        empty_mask = score_mask.eq(0).unsqueeze(0).expand_as(expand_score)
        expand_score.data.masked_fill_(empty_mask.data, -float('inf'))
        return expand_score.contiguous().view(batch_size, -1)  # batch x (context_len * context_len)

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

    def remove_bert_token(self, s):
        res = []
        for k in range(len(s)):
            if '##' in s[k] and k>0:
                res[-1] += s[k][2:]
            else:
                res.append(s[k])
        return res

    def f1_score(self, prediction, ground_truth):
        prediction_tokens = self.normalize_answer(prediction).split()
        ground_truth_tokens = self.normalize_answer(ground_truth).split()
        prediction_tokens = self.remove_bert_token(prediction_tokens)
        ground_truth_tokens = self.remove_bert_token(ground_truth_tokens)
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
            # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
            return int(ground_truth_tokens == prediction_tokens)
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

    def cal_rouge(self, infer, ref):
        x = rouge(infer, ref)
        return x['rouge_1/f_score'] * 100, x['rouge_2/f_score'] * 100, x['rouge_l/f_score'] * 100

    def rouge_max_over_ground_truths(self, prediction, ground_truths):
        scores_for_rouge1 = []
        scores_for_rouge2 = []
        scores_for_rougel = []
        for ground_truth in ground_truths:
            score = self.cal_rouge([prediction], [ground_truth])
            scores_for_rouge1.append(score[0])
            scores_for_rouge2.append(score[1])
            scores_for_rougel.append(score[2])
        return max(scores_for_rouge1), max(scores_for_rouge2), max(scores_for_rougel)

