# -- coding:UTF-8 --


import re
import os
import sys
import random
import math
import string
import logging
import argparse
import unicodedata
import collections
from shutil import copyfile
from datetime import datetime
from collections import Counter
from collections import defaultdict
import torch
import torch.nn as nn
import msgpack
import json
import numpy as np
import pandas as pd
from Models.Bert.tokenization import BertTokenizer
from Utils.GeneralUtils import normalize_text, nlp
from Utils.Constants import *
from torch.autograd import Variable
from Models.dataprocess import dataprocess, normalizeString


POS = {w: i for i, w in enumerate([''] + list(nlp.tagger.labels))}
ENT = {w: i for i, w in enumerate([''] + nlp.entity.move_names)}


def build_embedding(embed_file, targ_vocab, wv_dim):
    vocab_size = len(targ_vocab)
    emb = np.random.uniform(-1, 1, (vocab_size, wv_dim))  # 随机初始化一个词向量矩阵
    emb[0] = 0  # <PAD> should be all 0 (using broadcast)

    w2id = {w: i for i, w in enumerate(targ_vocab)}
    lineCnt = 0
    with open(embed_file, encoding="utf8") as f:
        for line in f:
            lineCnt = lineCnt + 1
            if lineCnt % 100000 == 0:
                print('.', end = '', flush=True)
            elems = line.split()
            token = normalize_text(''.join(elems[0:-wv_dim]))  # 找到glove中那个词
            if token in w2id:
                emb[w2id[token]] = [float(v) for v in elems[-wv_dim:]]  # 对找到的词记录embedding 向量
    return emb


def token2id_sent(sent, w2id, unk_id=None, to_lower=False):
    if to_lower:
        sent = sent.lower()
    w2id_len = len(w2id)
    ids = [w2id[w] if w in w2id else unk_id for w in sent]
    return ids


def char2id_sent(sent, c2id, unk_id=None, to_lower=False):
    if to_lower:
        sent = sent.lower()
    cids = [[c2id["<STA>"]] + [c2id[c] if c in c2id else unk_id for c in w] + [c2id["<END>"]] for w in sent]
    return cids


def token2id(w, vocab, unk_id=None):
    return vocab[w] if w in vocab else unk_id


'''
 Generate feature per context word according to its exact match with question words
'''


def feature_gen(context, question):
    counter_ = Counter(w.text.lower() for w in context)  # 统计每个单词出现的次数
    total = sum(counter_.values())
    term_freq = [counter_[w.text.lower()] / total for w in context]  # 计算TF
    question_word = {w.text for w in question}
    question_lower = {w.text.lower() for w in question}
    question_lemma = {w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower() for w in question}  # 问句每个词的原型
    match_origin = [w.text in question_word for w in context]  # 为文章中每个词判断是否出现在问题中 返回 true或false
    match_lower = [w.text.lower() in question_lower for w in context]  # 判断文章中每个词的小写是否出现在问题词小写词中,返回 true或false
    match_lemma = [(w.lemma_ if w.lemma_ != '-PRON-' else w.text.lower()) in question_lemma for w in
                   context]  # 判断文章中每个词的原型是否出现在问题词原型中，返回true或false
    C_features = list(zip(term_freq, match_origin, match_lower, match_lemma))
    return C_features


'''
 Get upper triangle matrix from start and end scores (batch)
 Input:
  score_s: batch x context_len
  score_e: batch x context_len
  context_len: number of words in context
  max_len: maximum span of answer
  use_cuda: whether GPU is used
 Output:
  expand_score: batch x (context_len * context_len) 
'''


def gen_upper_triangle(score_s, score_e, max_len, use_cuda):
    batch_size = score_s.shape[0]
    context_len = score_s.shape[1]
    # batch x context_len x context_len
    expand_score = score_s.unsqueeze(2).expand([batch_size, context_len, context_len]) + \
                   score_e.unsqueeze(1).expand([batch_size, context_len, context_len])
    score_mask = torch.ones(context_len)
    if use_cuda:
        score_mask = score_mask.cuda()
    score_mask = torch.ger(score_mask, score_mask).triu().tril(max_len - 1)
    empty_mask = score_mask.eq(0).unsqueeze(0).expand_as(expand_score)
    expand_score.data.masked_fill_(empty_mask.data, -float('inf'))
    return expand_score.contiguous().view(batch_size, -1)  # batch x (context_len * context_len)

SOS_token = 0
EOS_token = 1

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence, length):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    mask = [1 for _ in range(len(indexes))]
    while len(indexes) < length:
        indexes.append(0)
        mask.append(0)
    sentence_tensor = torch.tensor(indexes, dtype=torch.long).view(1,-1)
    mask_tensor = torch.tensor(mask, dtype=torch.long).view(1,-1)
    return sentence_tensor, mask_tensor

class BatchGen:
    def __init__(self, opt, data, lang, use_cuda, Is_training):
        # file_name = os.path.join(self.spacyDir, 'coqa-' + dataset_label + '-preprocessed.json')

        self.data = data
        self.lang = lang
        self.use_cuda = use_cuda
        self.opt = opt
        self.Is_training = Is_training
        self.bert_tokenizer = None
        if 'BERT' in self.opt:
            if 'BERT_LARGE' in opt:
                print('Using BERT Large model')
                tokenizer_file = os.path.join(opt['datadir'], opt['BERT_large_tokenizer_file'])
                print('Loading tokenizer from', tokenizer_file)
                self.bert_tokenizer = BertTokenizer.from_pretrained(tokenizer_file)
            else:
                print('Using BERT base model')
                if self.opt['dataset'] == 'coqa' or self.opt['dataset'] == 'quac':
                    tokenizer_file = os.path.join(opt['datadir'], opt['coqa_BERT_tokenizer_file'])
                    print('Loading tokenizer from', tokenizer_file)
                    self.bert_tokenizer = BertTokenizer.from_pretrained(tokenizer_file)
                else:
                    tokenizer_file = os.path.join(opt['datadir'], opt['BERT_tokenizer_file'])
                    print('Loading tokenizer from', tokenizer_file)
                    self.bert_tokenizer = BertTokenizer.from_pretrained(tokenizer_file)

        self.answer_span_in_context = 'ANSWER_SPAN_IN_CONTEXT_FEATURE' in self.opt
        self.ques_max_len = self.opt["ques_max_len"]
        self.ans_max_len = self.opt["ans_max_len"]
        self.doc_stride = self.opt["doc_stride"]
        print('*****************')
        print('ques_max_len     :', self.ques_max_len)
        print('ans_max_len      :', self.ans_max_len)
        print('sentence_max_len :', self.opt['max_featrue_length'])
        print('doc_stride       :', self.doc_stride)
        print('*****************')

        # random shuffle for training
        if Is_training:
            indices = list(range(len(self.data)))
            random.shuffle(indices)
            self.data = [self.data[i] for i in indices]  # 随机打乱数据

    def __len__(self):
        return len(self.data)

    def answer_bertify(self, words):
        if self.bert_tokenizer is None:
            return None
        res = []
        for w in words:
            now = self.bert_tokenizer.tokenize(w)
            res += now
        return res

    def q_bertify(self, words):
        if self.bert_tokenizer is None:
            return None

        bpe = ['[CLS]'] + words
        bpe.append('[SEP]')

        x_bert = self.bert_tokenizer.convert_tokens_to_ids(bpe)
        return x_bert

    def bertify(self, bpe, truth, span):
        '''
        先在bpe(文本)中检索span answer
        不存在： s, e 值溢出
        存在： s, e 为span answer在bpe的index
                                再在span answer检索 truth answer的 位置
        '''
        if self.bert_tokenizer is None:
            return None
        s, e = -math.inf, math.inf
        n, p1 = len(span), 0
        for t, word in enumerate(bpe):
            if bpe[t: t+n] == span:
                s, e = t, t+n
                if truth == span:
                    break
                l = len(truth)
                for k, w in enumerate(span):
                    if span[k: k+l] == truth:
                        s, e = s + k, s + k+l

        x_bert = self.bert_tokenizer.convert_tokens_to_ids(bpe)
        return x_bert, s, e


    def get_raw_context_offsets(self, words, raw_text):
        raw_context_offsets = []
        p = 0
        for token in words:
            while p < len(raw_text) and re.match('\s', raw_text[p]):
                p += 1
            if raw_text[p:p + len(token)] != token:
                # print('something is wrong! token', token, 'raw_text:', raw_text)
                print('something is wrong! token: ', token, p)

            raw_context_offsets.append((p, p + len(token)))
            p += len(token)

        return raw_context_offsets

    def __iter__(self):
        data = self.data
        MAX_ANS_SPAN = self.ans_max_len
        batch_size = self.opt['BATCH_SIZE']
        max_his_turns = self.opt['max_turn_nums']
        x_bert_list, x_bert_offsets_list, x_bert_mask_list, cur_q_id_list, cur_q_mask_list, \
        ground_truth_list, context_str_list, context_words_list, context_word_offsets_list,\
        span_answer_strs_list, max_context_list, answer_type_list, input_answer_strs_list, context_id_list, turn_id_list, his_inf_list = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

        for datum in data:
            # if self.Is_training:
                # remove super long answers for training
            datum['qas'] = [qa for qa in datum['qas'] if (len(qa['annotated_answer']['word']) == 1
                                            or qa['answer_span_end'] - qa['answer_span_start'] < MAX_ANS_SPAN)]
            # else:
            #     datum['qas'] = [qa for qa in datum['qas']]

            if len(datum['qas']) == 0:
                continue

            qa_len = len(datum['qas'])
            context_id = datum['id']

            for i in range(qa_len):
                if qa_len==1:
                    continue

                if self.opt['dataset'] == 'quac':
                    if datum['qas'][i]['turn_id'][-3:-1]=="q#":
                        turn_id = int(datum['qas'][i]['turn_id'][-1]) + 1
                else:
                    turn_id = int(datum['qas'][i]['turn_id'])
                if self.opt['dataset'] == 'coqa' or self.opt['dataset'] == 'quac':
                    cur_query_tokens = datum['qas'][i]['annotated_question']['word']
                else:
                    cur_query_tokens = datum['qas'][i]['question']
                cur_query_token = []
                for q in cur_query_tokens:
                    now = self.bert_tokenizer.tokenize(q)
                    cur_query_token += now
                cur_q_length = len(cur_query_token)

                if cur_q_length > self.ques_max_len:
                    cur_query_token = cur_query_token[ :self.ques_max_len]

                yes_no = []
                for w in ['yes', 'no', 'unknown']:
                    now = self.bert_tokenizer.tokenize(w)
                    yes_no += now
                ### 处理文章切片
                max_seq_length = self.opt['max_featrue_length']#384
                # 3是^，3是" yes no unknow"
                max_doc_length = max_seq_length - cur_q_length - 3 - len(yes_no)
                if self.opt['dataset'] == 'coqa' or self.opt['dataset'] == 'quac':
                    tokens = datum['annotated_context']['word']
                    all_doc_tokens = []
                    for t in tokens:
                        now = self.bert_tokenizer.tokenize(t)
                        all_doc_tokens += now
                else:
                    tokens = datum['context']

                _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
                    "DocSpan", ["start", "length"])
                doc_spans = []
                start_offset = 0
                while start_offset < len(all_doc_tokens):
                    length = len(all_doc_tokens) - start_offset
                    if length > max_doc_length:
                        length = max_doc_length
                    doc_spans.append(_DocSpan(start=start_offset, length=length))
                    if start_offset + length == len(all_doc_tokens):
                        break
                    start_offset += min(length, self.doc_stride)

                for (doc_index, doc_span) in enumerate(doc_spans):
                    ### 获取答案
                    if self.opt['dataset'] == 'coqa' or self.opt['dataset'] == 'quac':
                        span = datum['qas'][i]['span_text'].split(' ')
                        inputs = datum['qas'][i]['raw_answer'].split(' ')
                        span_answer_strs = self.answer_bertify(span)
                        input_answer_strs = self.answer_bertify(inputs)
                        answer_type = datum['qas'][i]['answer_type']
                    elif self.opt['dataset'] == 'chinese':
                        span_answer_strs = datum['qas'][i]['answer']
                        input_answer_strs = datum['qas'][i]['answer']
                        answer_type = datum['qas'][i]['answer_type']


                    ### 形成 [CLS] + Q + [SEP] + P + [SEP]
                    if self.opt['dataset'] == 'coqa' or self.opt['dataset'] == 'quac':
                        seq = ['[CLS]'] + cur_query_token + ['[SEP]'] + all_doc_tokens[doc_span.start: doc_span.start+doc_span.length] + yes_no + ['[SEP]']
                        cur_q = cur_query_token
                    else:
                        seq = [cur_query_token] + ['^'] + [all_doc_tokens[doc_span.start: doc_span.start+doc_span.length]]
                        cur_q = [cur_query_token]

                    pre_length = len(cur_q) + 2
                    # cur_q = normalizeString(datum['qas'][i]['question'])
                    if len(cur_q) > self.ques_max_len:
                        cur_q = cur_q[:self.ques_max_len]
                        tmp = len(cur_q) - 1
                        while cur_q[tmp] != " ":
                            tmp -= 1
                        cur_q = cur_q[:tmp + 1]

                    # cur_q_id, cur_q_mask = tensorFromSentence(self.lang, cur_q, self.ques_max_len)
                    cur_q = self.q_bertify(cur_q)
                    cur_q_mask = torch.LongTensor(1, len(cur_q)).fill_(1)
                    cur_q = torch.tensor([cur_q], dtype=torch.long, requires_grad=False)
                    if len(cur_q[0]) < self.ques_max_len:
                        num = self.ques_max_len - len(cur_q[0])
                        pad = nn.ZeroPad2d(padding=(0, num, 0, 0))
                        cur_q_id = pad(cur_q)
                        cur_q_mask = pad(cur_q_mask)

                    x_bert, start, end = self.bertify(seq, input_answer_strs, span_answer_strs)
                    x_bert_mask = torch.LongTensor(1, len(x_bert)).fill_(1)
                    x_bert = torch.tensor([x_bert], dtype=torch.long)
                    x_bert_offsets = [1]
                    x_bert_offsets = torch.tensor([x_bert_offsets], dtype=torch.long)
                    # print("x_bert: ", x_bert.size())
                    # print("x_bert_mask: ", x_bert_mask.size())

                    if self.opt['dataset'] == 'coqa' or self.opt['dataset'] == 'quac':
                        # context_str = ['CLS'] + cur_query_token + ['SEP'] + all_doc_tokens[doc_span.start: doc_span.start + doc_span.length] + ['yes', 'no', 'unknown'] + ['SEP']
                        context_str = seq
                        # context_str = "".join(context_str)
                    else:
                        context_str = ['^'] + [cur_query_token] + ['^'] + [all_doc_tokens[doc_span.start: doc_span.start+doc_span.length]] + ['^']
                        # context_str = "".join(context_str)

                    if len(x_bert[0]) < max_seq_length:
                        num = max_seq_length - len(x_bert[0])
                        pad = nn.ZeroPad2d(padding=(0, num, 0, 0))
                        x_bert = pad(x_bert)
                        x_bert_mask = pad(x_bert_mask)

                    token_is_max_context = {}
                    for index in range(doc_span.length):
                        split_token_index = doc_span.start + index
                        is_max_context = _check_is_max_context(doc_spans, doc_index,
                                                               split_token_index)
                        token_is_max_context[pre_length] = is_max_context
                        pre_length += 1
                    context_word_offsets = datum['raw_context_offsets'][doc_span.start: doc_span.start+doc_span.length]
                    context_words = datum['annotated_context']['word'][doc_span.start: doc_span.start+doc_span.length]
                    tok_start_position = datum['qas'][i]['answer_span_start']
                    tok_end_position = datum['qas'][i]['answer_span_end']

                    ### 给yes no unknow加上offset
                    end_pos = context_word_offsets[-1][-1]
                    context_word_offsets += [[end_pos+1, end_pos+4]] + [[end_pos+5, end_pos+7]] + [[end_pos+8, end_pos+15]]
                    tok_start_position = start
                    tok_end_position = end

                    tmp = input_answer_strs[0]
                    l = len(context_str)
                    if tmp == 'yes':
                        tok_start_position = l - 4
                        tok_end_position = l - 3
                    elif tmp == 'no':
                        tok_start_position = l - 3
                        tok_end_position = l - 2
                    elif tmp == 'unknown':
                        tok_start_position = l - 2
                        tok_end_position = l -1

                    print("lost: ", input_answer_strs)
                    print(tok_start_position, tok_end_position)
                    if self.Is_training:
                        out_of_span = False
                        if self.opt['dataset'] == 'coqa' or self.opt['dataset'] == 'quac':
                            if tok_start_position<0 or tok_end_position>max_seq_length:
                                out_of_span = True
                        else:
                            if answer in all_doc_tokens[doc_span.start: doc_span.start+doc_span.length]:
                                tok_start_position = all_doc_tokens.index(answer)
                                tok_end_position = tok_start_position + len(answer)
                            else:
                                out_of_span = True


                        ### 产生的文章变体中，如果不包含答案，那么会跳过该片段
                        if out_of_span:
                            # start_position = 0
                            # end_position = 0
                            continue
                        else:
                            start_position = tok_start_position
                            end_position = tok_end_position

                    else:
                        # when predicting, we do not throw out any doc span to prevent label leaking

                        if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                            start_position, end_position = None, None
                        else:
                            start_position = tok_start_position
                            end_position = tok_end_position

                    # answer
                    ground_truth = [start_position, end_position]

                    # if ground_truth[0] == None:
                    #     answer_str = None
                    # elif ground_truth[0] >= 0 and ground_truth[1] >= 0:
                    #     answer_str = answer
                    # # answer_strs = []
                    # all_viable_answers = [answer_str]
                    # # if 'additional_answers' in datum['qas'][i]:
                    # #     all_viable_answers.extend(datum['qas'][i]['additional_answers'])


                    if self.use_cuda:
                        x_bert = Variable(x_bert.cuda())
                        x_bert_mask = Variable(x_bert_mask.cuda())
                        cur_q_id = Variable(cur_q_id.cuda())
                        cur_q_mask = Variable(cur_q_mask.cuda())
                    else:
                        x_bert = Variable(x_bert)
                        x_bert_mask = Variable(x_bert_mask)
                        cur_q_id = Variable(cur_q_id)
                        cur_q_mask = Variable(cur_q_mask)

                    x_bert_list.append(x_bert)
                    x_bert_offsets_list.append(x_bert_offsets)
                    x_bert_mask_list.append(x_bert_mask)
                    cur_q_id_list.append(cur_q_id)
                    cur_q_mask_list.append(cur_q_mask)
                    ground_truth_list.append(ground_truth)
                    context_str_list.append(context_str)
                    context_words_list.append(context_words)
                    context_word_offsets_list.append(context_word_offsets)
                    span_answer_strs_list.append(span_answer_strs)
                    max_context_list.append(token_is_max_context)
                    answer_type_list.append(answer_type)
                    input_answer_strs_list.append(input_answer_strs)
                    context_id_list.append(context_id)
                    turn_id_list.append(turn_id)

                    his_list = []
                    his_mask_list = []
                    his_inf = {}
                    qa_pair = []
                    for j in range(i-max_his_turns, i):
                        if j<0:
                            continue
                        # if self.opt['dataset'] == 'coqa':
                        # query_token = normalizeString(datum['qas'][j]['question'])
                        # answer_token = normalizeString(datum['qas'][j]['answer'])
                        query_token = self.answer_bertify(datum['qas'][j]['question'].split(' '))
                        answer_token = self.answer_bertify(datum['qas'][j]['answer'].split(' '))
                        if len(query_token) > self.ques_max_len:
                            query_token = query_token[ :self.ques_max_len]
                            tmp = len(query_token) - 1
                            while query_token[tmp]!=" ":
                                tmp -= 1
                            query_token = query_token[:tmp]
                        if len(answer_token) > self.ans_max_len:
                            answer_token = answer_token[ :self.ans_max_len]
                            tmp = len(answer_token) - 1
                            while answer_token[tmp] != " ":
                                tmp -= 1
                            answer_token = answer_token[:tmp]
                        qa_pair += query_token + ['[SEP]'] + answer_token + ['[SEP]']
                    if qa_pair:
                        qa_pair = ['[CLS]'] + qa_pair
                    # 将历史信息拼成了一句话 qaqaqa 可以加个分隔符号
                    qa_max_len = (self.ques_max_len + self.ans_max_len + 1) * max_his_turns + max_his_turns - 1
                    # his, his_bert_mask = tensorFromSentence(self.lang, qa_pair, qa_max_len)

                    if qa_pair:
                        his = self.bert_tokenizer.convert_tokens_to_ids(qa_pair)
                        # his, his_offsets, bpe, _ = self.bertify(qa_pair)
                        his_bert_mask = torch.LongTensor(1, len(his)).fill_(1)
                        his = torch.tensor([his], dtype = torch.long, requires_grad=False)
                        if len(his[0]) < qa_max_len:
                            num = qa_max_len - len(his[0])
                            pad = nn.ZeroPad2d(padding=(0, num, 0, 0))
                            his = pad(his)
                            his_bert_mask = pad(his_bert_mask)
                    else:
                        his = torch.zeros([1, qa_max_len], dtype = torch.long)
                        his_bert_mask = torch.ones([1,qa_max_len], dtype = torch.long)


                    if self.use_cuda:
                        his = Variable(his.cuda())
                        his_bert_mask = Variable(his_bert_mask.cuda())
                    else:
                        his = Variable(his)
                        his_bert_mask = Variable(his_bert_mask)

                    his_list.append(his)
                    his_mask_list.append(his_bert_mask)
                    his_inf["his"] = his_list
                    his_inf["his_mask"] = his_mask_list
                    his_inf_list.append(his_inf)

                    print("answer_type: ", answer_type)
                    # print("ground_truth: ", ground_truth)
                    print("length: ", len(context_str), x_bert.size(1))
                    print("truth: ", " ".join(input_answer_strs))
                    print("truth: ", " ".join(context_str[ground_truth[0] : ground_truth[1]]))
                    # print("truth: ", " ".join(context_str[ground_truth[0] : ground_truth[1]]))
                    # if sign: print(sign)

                    # print(context_str)
                    # print(len(context_str))
                    #
                    # print(context_word_offsets)
                    # print(len(context_word_offsets))
                    # print("cur_q: ", q_print)
                    if qa_pair:
                        print("qa_pair: ", qa_pair)
                    print("\n")
                        ### 输出当前Q+P拼它所有history
                    if len(x_bert_list)==batch_size:
                        yield (
                            x_bert_list, x_bert_offsets_list, x_bert_mask_list, cur_q_id_list, cur_q_mask_list,
                            ground_truth_list, context_str_list, context_words_list,
                            context_word_offsets_list, span_answer_strs_list, max_context_list, answer_type_list, input_answer_strs_list, context_id_list, turn_id_list, his_inf_list)
                        exit(0)
                        x_bert_list, x_bert_offsets_list, x_bert_mask_list, cur_q_id_list, cur_q_mask_list, \
                        ground_truth_list, context_str_list, context_words_list, context_word_offsets_list, \
                        span_answer_strs_list, max_context_list, answer_type_list, input_answer_strs_list, context_id_list, turn_id_list, his_inf_list = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

        # # 最后的list里，数量不为batch_size, 但是已经到最后一步
        # if x_bert_list:
        #     yield (
        #         x_bert_list, x_bert_offsets_list, x_bert_mask_list, cur_q_bert_list, q_bert_mask_list,
        #         ground_truth_list, context_str_list, context_words_list,
        #         context_word_offsets_list, answer_strs_list, context_id_list, turn_id_list, his_inf_list)

def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


# ===========================================================================
# =================== For standard evaluation in CoQA =======================
# ===========================================================================

def ensemble_predict(pred_list, score_list, voteByCnt=False):
    predictions, best_scores = [], []
    pred_by_examples = list(zip(*pred_list))
    score_by_examples = list(zip(*score_list))
    for phrases, scores in zip(pred_by_examples, score_by_examples):
        d = defaultdict(float)
        firstappear = defaultdict(int)
        for phrase, phrase_score, index in zip(phrases, scores, range(len(scores))):
            d[phrase] += 1. if voteByCnt else phrase_score
            if not phrase in firstappear:
                firstappear[phrase] = -index
        predictions += [max(d.items(), key=lambda pair: (pair[1], firstappear[pair[0]]))[0]]
        best_scores += [max(d.items(), key=lambda pair: (pair[1], firstappear[pair[0]]))[1]]
    return (predictions, best_scores)


def _f1_score(pred, answers):
    def _score(g_tokens, a_tokens):
        common = Counter(g_tokens) & Counter(a_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1. * num_same / len(g_tokens)
        recall = 1. * num_same / len(a_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    if pred is None or answers is None:
        return 0

    if len(answers) == 0:
        return 1. if len(pred) == 0 else 0.

    g_tokens = _normalize_answer(pred).split()
    ans_tokens = [_normalize_answer(answer).split() for answer in answers]
    scores = [_score(g_tokens, a) for a in ans_tokens]
    if len(ans_tokens) == 1:
        score = scores[0]
    else:
        score = 0
        for i in range(len(ans_tokens)):
            scores_one_out = scores[:i] + scores[(i + 1):]
            score += max(scores_one_out)
        score /= len(ans_tokens)
    return score


def score(pred, truth, final_json):
    assert len(pred) == len(truth)
    no_ans_total = no_total = yes_total = normal_total = total = 0
    no_ans_f1 = no_f1 = yes_f1 = normal_f1 = f1 = 0
    all_f1s = []
    for p, t, j in zip(pred, truth, final_json):
        total += 1
        this_f1 = _f1_score(p, t)
        f1 += this_f1
        all_f1s.append(this_f1)
        if t[0].lower() == 'no':
            no_total += 1
            no_f1 += this_f1
        elif t[0].lower() == 'yes':
            yes_total += 1
            yes_f1 += this_f1
        elif t[0].lower() == 'unknown':
            no_ans_total += 1
            no_ans_f1 += this_f1
        else:
            normal_total += 1
            normal_f1 += this_f1

    f1 = 100. * f1 / total
    if no_total == 0:
        no_f1 = 0.
    else:
        no_f1 = 100. * no_f1 / no_total
    if yes_total == 0:
        yes_f1 = 0
    else:
        yes_f1 = 100. * yes_f1 / yes_total
    if no_ans_total == 0:
        no_ans_f1 = 0.
    else:
        no_ans_f1 = 100. * no_ans_f1 / no_ans_total
    normal_f1 = 100. * normal_f1 / normal_total
    result = {
        'total': total,
        'f1': f1,
        'no_total': no_total,
        'no_f1': no_f1,
        'yes_total': yes_total,
        'yes_f1': yes_f1,
        'no_ans_total': no_ans_total,
        'no_ans_f1': no_ans_f1,
        'normal_total': normal_total,
        'normal_f1': normal_f1,
    }
    return result, all_f1s


def score_each_instance(pred, truth):
    assert len(pred) == len(truth)
    total = 0
    f1_scores = []
    for p, t in zip(pred, truth):
        total += 1
        f1_scores.append(_f1_score(p, t))
    f1_scores = [100. * x / total for x in f1_scores]
    return f1_scores


def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        if type(text)==list:
            new = []
            for i in range(len(text)):
                new.append(text[i].lower())
            text = new
        else:
            text = text.lower()
        return text

    return white_space_fix(remove_articles(remove_punc(lower(s))))


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes

def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, for_reward=False):
    """Write final predictions to the json file."""
    tf.logging.info("Writing predictions to: %s" % (output_prediction_file))
    tf.logging.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit", "predicted_yesno",
         "predicted_followup"])

    yesno_dict = ['y', 'n', 'x']
    followup_dict = ['y', 'n', 'm']

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]
        # if len(features) == 0:
        #    continue

        prelim_predictions = []
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]

            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            predicted_yesno = yesno_dict[np.argmax(result.yesno_logits)]
            predicted_followup = followup_dict[np.argmax(result.followup_logits)]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                            predicted_yesno=predicted_yesno,
                            predicted_followup=predicted_followup
                        ))

        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit", "predicted_yesno", "predicted_followup"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]

            tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
            orig_doc_start = feature.token_to_orig_map[pred.start_index]
            orig_doc_end = feature.token_to_orig_map[pred.end_index]
            orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
            tok_text = " ".join(tok_tokens)

            # De-tokenize WordPieces that have been split off.
            tok_text = tok_text.replace(" ##", "")
            tok_text = tok_text.replace("##", "")

            # Clean whitespace
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            orig_text = " ".join(orig_tokens)

            final_text = get_final_text(tok_text, orig_text, do_lower_case)
            if final_text in seen_predictions:
                continue

            seen_predictions[final_text] = True
            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit,
                    predicted_yesno=pred.predicted_yesno,
                    predicted_followup=pred.predicted_followup
                ))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            if for_reward:
                continue
            if FLAGS.dataset.lower() == 'coqa':
                nbest.append(_NbestPrediction(text="unknown", start_logit=0.0, end_logit=0.0))
            elif FLAGS.dataset.lower() == 'quac':
                nbest.append(_NbestPrediction(text="invalid", start_logit=0.0, end_logit=0.0, predicted_yesno='y',
                                              predicted_followup='y'))

        assert len(nbest) >= 1

        total_scores = []
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            output['yesno'] = entry.predicted_yesno
            output['followup'] = entry.predicted_followup
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        all_predictions[example.qas_id] = (nbest_json[0]["text"], nbest_json[0]['yesno'], nbest_json[0]['followup'])
        all_nbest_json[example.qas_id] = nbest_json
