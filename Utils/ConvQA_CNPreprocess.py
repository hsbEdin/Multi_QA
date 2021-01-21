# -- coding:UTF-8 --

import json
import msgpack
import multiprocessing
import re
import string
import torch
import jieba
import unicodedata
import numpy as np
from tqdm import tqdm
from collections import Counter
import Utils.tokenization_word as tokenization
from Utils.GeneralUtils import nlp, load_word2vec_vocab, pre_proc

# from Utils.CoQAUtils import token2id, token2id_sent, char2id_sent, build_embedding, feature_gen, POS, ENT
import os

class ConvQA_CNPreprocess():
    def __init__(self, opt):
        print("ConvQA_CN Preprocessing...")
        self.opt = opt
        self.spacyDir = opt['FEATURE_FOLDER']
        self.train_file = os.path.join(opt['datadir'], opt['TRAIN_FILE'])
        self.dev_file = os.path.join(opt['datadir'], opt['DEV_FILE'])
        self.word2vec_file = os.path.join(opt['datadir'], opt['WORD_EMBEDDING_FILE'])
        self.word2vec_dim = 300
        print("The path of train file: ", self.train_file)
        print("The path of dev file: ", self.dev_file)
        print("The path of FEATURE_FOLDER: ", self.spacyDir)
        self.data_prefix = 'ConvQA_CN-'

        dataset_labels = ['train', 'dev']
        allExist = True
        for dataset_label in dataset_labels:
            if not os.path.exists(os.path.join(self.spacyDir, self.data_prefix + dataset_label + '-preprocessed.json')):
                allExist = False

        #如果已经有features，怎么不需要做预处理
        if allExist:
            return

        print('Previously result not found, creating preprocessed files now...')
        # self.word2vec_vocab = load_word2vec_vocab(self.word2vec_file, self.word2vec_dim)
        if not os.path.isdir(self.spacyDir):
            os.makedirs(self.spacyDir)
            print('Directory created: ' + self.spacyDir)

        for dataset_label in dataset_labels:
            self.preprocess(dataset_label)

    def remove_punctual(self, s):
        s = s.replace(",", "")
        s = s.replace(".", "")
        s = s.replace("?", "")
        s = s.replace("!", "")

        return s

    def whitespace(self, context_str):
        context_str = context_str.replace("\u3000", "")
        context_str = context_str.replace("\n", "")
        context_str = context_str.replace("\t", "")
        context_str = context_str.replace("\r", "")
        context_str = context_str.replace("\xa0", "")
        context_str = context_str.replace(" ", "")
        return context_str

    def preprocess(self, dataset_label):
        file_name = self.train_file if dataset_label == 'train' else (self.dev_file if dataset_label == 'dev' else self.test_file)
        output_file_name = os.path.join(self.spacyDir, self.data_prefix + dataset_label + '-preprocessed.json')
        print('Preprocessing', dataset_label, 'file:', file_name)
        print('The output file is in: ', output_file_name)
        print('Loading json...')
        with open(file_name, 'r') as f:
            dataset = json.load(f)

        print('Processing json...')
        data = []
        tot = len(dataset['data'])
        print(tot)

        dict1 = ['where', 'when', 'who']

        for data_idx in tqdm(range(tot)):
            datum = dataset['data'][data_idx]
            context_str = datum['Passage']
            context_str = self.whitespace(context_str)
            context = ''.join(jieba.cut(context_str, cut_all=False))
            context = context + "无法作答。"
            _datum = {'context': context,
                      'source': datum['Source'],
                      'id': datum['id']}

            _datum['annotated_context'] = {}
            #这里得到了分词，存在_datum['annotated_context']['word']中
            _datum['annotated_context']['word'] = ' '.join(context_str).split()

            _datum['raw_context_offsets'] = self.get_raw_context_offsets(_datum['annotated_context']['word'],
                                                                         context_str)

            # print(_datum['raw_context_offsets'])

            _datum['qas'] = []
            # assert len(datum['Questions']) <= len(datum['Answers'])
            if len(datum['Questions']) <= len(datum['Answers']):
                turn_range = len(datum['Questions'])
            else:
                turn_range = len(datum['Answers'])
            """ additional answer can be eval data
            additional_answers = {}
            """
            j = 0
            print("文章id: ", _datum['id'])
            for i in range(turn_range):
                question = datum['Questions'][i]
                tmp_id = question['turn_id']

                ### 取出问题对应的所有答案中，第一个答案
                print("i: {}, j: {}".format(i, j))
                while datum['Answers'][j]['turn_id']!=tmp_id:
                    j+=1
                    if j > i:
                        continue
                    # print("j: ", j)
                answer = datum['Answers'][j]
                j+=1
                assert question['turn_id'] == answer['turn_id']

                question_text = "".join(jieba.cut(question['input_text'], cut_all=False)).strip()
                question_text = self.whitespace(question_text)
                answer_text = "".join(jieba.cut(answer['input_text'], cut_all=False)).strip()
                answer_text = self.whitespace(answer_text)
                idx = question['turn_id']
                _qas = {'turn_id': idx,
                        'question': question_text,
                        'answer': answer_text}
                ### 如果有额外回答，需要在这里做处理
                _qas['annotated_question'] = {}
                _qas['annotated_answer'] = {}

                _qas['annotated_question']['word'] = ' '.join(jieba.cut(question['input_text'], cut_all=False)).split()

                _qas['annotated_answer']['word'] = ' '.join(jieba.cut(answer['input_text'], cut_all=False)).split()
                _qas['raw_answer'] = answer['input_text']
                _qas['span_text'] = answer['input_text']
                _qas['answer_type'] = answer['answer_type']

                sign = ""
                ques = question['input_text'].lower()
                real_ans = answer['input_text'].lower()
                real = self.remove_punctual(real_ans)
                real = real.split()

                for word in dict1:
                    if word in ques or ques[:3] == "was":
                        sign = "factual"
                        break

                if len(real) <= 4:
                    sign = "factual"
                if not sign or real_ans == "no" or real_ans == "yes":
                    sign = "non-factual"

                _qas['question_type'] = sign

                if _qas['answer_type'] == 'extractive':
                    start = _datum["context"].index(_qas["answer"])

                    end = start + len(_qas["answer"])

                    _qas['answer_span_start'], _qas['answer_span_end'] = start, end

                    chosen_text = _datum['context'][start: end]
                    while len(chosen_text) > 0 and chosen_text[0] in string.whitespace:
                        chosen_text = chosen_text[1:]
                        start += 1
                    while len(chosen_text) > 0 and chosen_text[-1] in string.whitespace:
                        chosen_text = chosen_text[:-1]
                        end -= 1
                    input_text = _qas['answer'].strip()
                    ### 这里相当于验证答案span是否正确
                    if input_text in chosen_text:
                        p = chosen_text.find(input_text)
                        _qas['answer_span'] = self.find_span(_datum['raw_context_offsets'],
                                                             start + p, start + p + len(input_text))
                    else:
                        _qas['answer_span'] = self.find_span_with_gt(_datum['context'],
                                                                     _datum['raw_context_offsets'], input_text)
                    ### 如果需要对当前问题答案扩充成 前几轮拼接， 需在这里添加代码


                _datum['qas'].append(_qas)
            data.append(_datum)

        dataset['data'] = data

        ### dataset_label == 'test'

        with open(output_file_name, 'w') as f:
            json.dump(dataset, f, sort_keys=True, indent=4)


    def load_data(self):
        print('Loading train_meta.msgpack...')
        meta_file_name = os.path.join(self.spacyDir, 'train_meta.msgpack')
        with open(meta_file_name, 'rb') as f:
            meta = msgpack.load(f)
        embedding = torch.Tensor(meta['embedding'])
        self.opt['vocab_size'] = embedding.size(0)
        self.opt['vocab_dim'] = embedding.size(1)
        return meta['vocab'], embedding

    def process(self, parsed_text):
        output = {'word': [],
                  'pos': [],
                  'pos_id': [],
                  'ent': [],
                  'ent_id': [],
                  'offsets': [],
                  'sentences': []}

        for token in parsed_text:
            output['word'].append(token.text)
            pos = token.tag_
            output['pos'].append(pos)
            output['pos_id'].append(token2id(pos, POS, 0))

            ent = 'O' if token.ent_iob_ == 'O' else (token.ent_iob_ + '-' + token.ent_type_)
            output['ent'].append(ent)
            output['ent_id'].append(token2id(ent, ENT, 0))

            output['lemma'].append(token.lemma_ if token.lemma_ != '-PRON-' else token.text.lower())
            output['offsets'].append((token.idx, token.idx + len(token.text)))

        word_idx = 0
        for sent in parsed_text.sents:
            output['sentences'].append((word_idx, word_idx + len(sent)))
            word_idx += len(sent)

        assert word_idx == len(output['word'])
        return output
        return output

    def _str(self, s):
        """ Convert PTB tokens to normal tokens """
        if (s.lower() == '-lrb-'):
            s = '('
        elif (s.lower() == '-rrb-'):
            s = ')'
        elif (s.lower() == '-lsb-'):
            s = '['
        elif (s.lower() == '-rsb-'):
            s = ']'
        elif (s.lower() == '-lcb-'):
            s = '{'
        elif (s.lower() == '-rcb-'):
            s = '}'
        return s

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

    def normalize_answer(self, s):
        """Lower text and remove punctuation, storys and extra whitespace."""

        def remove_articles(text):
            regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
            return re.sub(regex, ' ', text)

        def white_space_fix(text):
            return ' '.join(text.split())

        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)

        return white_space_fix(remove_articles(remove_punc(s)))

    def find_span_with_gt(self, context, offsets, ground_truth):
        best_f1 = 0.0
        best_span = (len(offsets) - 1, len(offsets) - 1)
        gt = self.normalize_answer(pre_proc(ground_truth)).split()

        ls = [i for i in range(len(offsets)) if context[offsets[i][0]:offsets[i][1]] in gt]

        for i in range(len(ls)):
            for j in range(i, len(ls)):
                pred = self.normalize_answer(pre_proc(context[offsets[ls[i]][0]: offsets[ls[j]][1]])).split()
                common = Counter(pred) & Counter(gt)
                num_same = sum(common.values())
                if num_same > 0:
                    precision = 1.0 * num_same / len(pred)
                    recall = 1.0 * num_same / len(gt)
                    f1 = (2 * precision * recall) / (precision + recall)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_span = (ls[i], ls[j])
        return best_span

    def find_span(self, offsets, start, end):
        start_index = -1
        end_index = -1
        for i, offset in enumerate(offsets):
            if (start_index < 0) or (start >= offset[0]):
                start_index = i
            if (end_index < 0) and (end <= offset[1]):
                end_index = i
        return (start_index, end_index)

