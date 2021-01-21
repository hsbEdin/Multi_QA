# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
    This file takes a CoQA data file as input and generates the input files for training the QA model.
"""
import json
import os
import msgpack
import multiprocessing
import re
import string
import torch
from tqdm import tqdm
from collections import Counter
from Utils.GeneralUtils import nlp, load_glove_vocab, pre_proc
from Utils.CoQAUtils import token2id, token2id_sent, char2id_sent, build_embedding, feature_gen, POS, ENT
import os


class QuACPreprocess():
    def __init__(self, opt):
        print('CoQA Preprocessing')
        self.opt = opt
        self.spacyDir = opt['FEATURE_FOLDER']
        self.train_file = os.path.join(opt['datadir'], opt['Quac_TRAIN_FILE'])
        self.dev_file = os.path.join(opt['datadir'], opt['Quac_DEV_FILE'])
        self.glove_file = os.path.join(opt['datadir'], opt['INIT_WORD_EMBEDDING_FILE'])
        self.glove_dim = 300
        print("The path of dev file: ", self.dev_file)
        print("The path of FEATURE_FOLDER: ", self.spacyDir)
        self.data_prefix = 'quac-'

        dataset_labels = ['train', 'dev']
        allExist = True
        for dataset_label in dataset_labels:
            if not os.path.exists(os.path.join(self.spacyDir, self.data_prefix + dataset_label + '-preprocessed.json')):
                allExist = False

        if allExist:
            return

        print('Previously result not found, creating preprocessed files now...')
        self.glove_vocab = load_glove_vocab(self.glove_file, self.glove_dim, to_lower=False)
        if not os.path.isdir(self.spacyDir):
            os.makedirs(self.spacyDir)
            print('Directory created: ' + self.spacyDir)

        for dataset_label in dataset_labels:
            self.preprocess(dataset_label)

    # dataset_label can be 'train' or 'dev' or 'test'
    def preprocess(self, dataset_label):
        file_name = self.train_file if dataset_label == 'train' else (
            self.dev_file if dataset_label == 'dev' else self.test_file)
        output_file_name = os.path.join(self.spacyDir, self.data_prefix + dataset_label + '-preprocessed.json')

        print('Preprocessing', dataset_label, 'file:', file_name)
        print('Loading json...')
        with open(file_name, 'r') as f:
            dataset = json.load(f)

        print('Processing json...')

        data = []
        tot = len(dataset['data'])
        for data_idx in tqdm(range(tot)):
            datum = dataset['data'][data_idx]['paragraphs'][0]
            context_str = datum['context']
            _datum = {'context': context_str,
                      'title': dataset['data'][data_idx]['title'],
                      'id': data_idx}

            nlp_context = nlp(pre_proc(context_str))
            _datum['annotated_context'] = self.process(nlp_context)
            _datum['raw_context_offsets'] = self.get_raw_context_offsets(_datum['annotated_context']['word'],
                                                                         context_str)
            _datum['qas'] = []

            # assert len(datum['qas']['questions']) == len(datum['answers'])

            for i in range(len(datum['qas'])):
                question, answer = datum['qas'][i]['question'], datum['qas'][i]['answers'][0]['text']
                # assert question['turn_id'] == answer['turn_id']

                idx = datum['qas'][i]['id']
                _qas = {'turn_id': idx,
                        'question': question,
                        'answer': answer}

                _qas['annotated_question'] = self.process(nlp(pre_proc(question)))

                _qas['annotated_answer'] = self.process(nlp(pre_proc(answer)))
                _qas['raw_answer'] = answer
                _qas['span_text'] = answer
                _qas['answer_type'] = "extractive"
                _qas['answer_span_start'] = datum['qas'][i]['answers'][0]['answer_start']
                _qas['answer_span_end'] = _qas['answer_span_start'] + len(answer) + 1

                start = _qas['answer_span_start']  # rational 范围
                end = _qas['answer_span_end']
                chosen_text = _datum['context'][start: end].lower()
                while len(chosen_text) > 0 and chosen_text[0] in string.whitespace:  # 判断开头的空白符 \t,\n等6种
                    chosen_text = chosen_text[1:]
                    start += 1
                while len(chosen_text) > 0 and chosen_text[-1] in string.whitespace:  # 判断结尾的空白符
                    chosen_text = chosen_text[:-1]
                    end -= 1
                input_text = _qas['answer'].strip().lower()
                if input_text in chosen_text:
                    p = chosen_text.find(input_text)  # p:input_text的起始值
                    _qas['answer_span'] = self.find_span(_datum['raw_context_offsets'],
                                                         start + p, start + p + len(input_text))
                else:
                    _qas['answer_span'] = self.find_span_with_gt(_datum['context'],
                                                                 _datum['raw_context_offsets'], input_text)

                _datum['qas'].append(_qas)

            data.append(_datum)

        # build vocabulary
        if dataset_label == 'train':
            print('Build vocabulary from training data...')
            contexts = [_datum['annotated_context']['word'] for _datum in data]
            qas = [qa['annotated_question']['word'] + qa['annotated_answer']['word'] for qa in _datum['qas'] for _datum
                   in data]
            self.train_vocab = self.build_vocab(contexts, qas)

        print('Getting word ids...')
        w2id = {w: i for i, w in enumerate(self.train_vocab)}
        for _datum in data:
            _datum['annotated_context']['wordid'] = token2id_sent(_datum['annotated_context']['word'], w2id, unk_id=1,
                                                                  to_lower=False)
            # new modify, get wordid
            for qa in _datum['qas']:
                qa['annotated_question']['wordid'] = token2id_sent(qa['annotated_question']['word'], w2id, unk_id=1,
                                                                   to_lower=False)
                qa['annotated_answer']['wordid'] = token2id_sent(qa['annotated_answer']['word'], w2id, unk_id=1,
                                                                 to_lower=False)

        if dataset_label == 'train':
            # get the condensed dictionary embedding
            print('Getting embedding matrix for ' + dataset_label)
            embedding = build_embedding(self.glove_file, self.train_vocab, self.glove_dim)
            meta = {'vocab': self.train_vocab, 'embedding': embedding.tolist()}
            meta_file_name = os.path.join(self.spacyDir, dataset_label + '_meta.msgpack')
            print('Saving meta information to', meta_file_name)
            with open(meta_file_name, 'wb') as f:
                msgpack.dump(meta, f, encoding='utf8')

        dataset['data'] = data

        if dataset_label == 'test':
            return dataset

        with open(output_file_name, 'w') as output_file:
            json.dump(dataset, output_file, sort_keys=True, indent=4)

    '''
     Return train_vocab embedding
    '''

    def load_data(self):
        print('Load train_meta.msgpack...')
        meta_file_name = os.path.join(self.spacyDir, 'train_meta.msgpack')
        with open(meta_file_name, 'rb') as f:
            meta = msgpack.load(f, encoding='utf8')
        embedding = torch.Tensor(meta['embedding'])
        self.opt['vocab_size'] = embedding.size(0)
        self.opt['vocab_dim'] = embedding.size(1)
        return meta['vocab'], embedding

    def build_vocab(self, contexts, qas):  # vocabulary will also be sorted accordingly
        counter_c = Counter(w for doc in contexts for w in doc)  # 文章词频统计
        counter_qa = Counter(w for doc in qas for w in doc)  # 问答词频统计，字典形式
        counter = counter_c + counter_qa  # 合并相同的字符统计值，字典类型
        vocab = sorted([t for t in counter_qa if t in self.glove_vocab], key=counter_qa.get,
                       reverse=True)  # 对统计的问答字符进行排序（在glove中的）
        vocab += sorted([t for t in counter_c.keys() - counter_qa.keys() if t in self.glove_vocab],
                        key=counter.get, reverse=True)  # 对保留下来的不同字符(文章和问答不同的字符，即文章中有，问答中已经统计过的在上一行已经统计过的再进行统计)进行排序
        total = sum(counter.values())  # 问答中的字符总数
        matched = sum(counter[t] for t in vocab)  # glove中存在的字符总数
        print('vocab {1}/{0} OOV {2}/{3} ({4:.4f}%)'.format(
            len(counter), len(vocab), (total - matched), total, (total - matched) / total * 100))
        vocab.insert(0, "<PAD>")
        vocab.insert(1, "<UNK>")
        vocab.insert(2, "<Q>")
        vocab.insert(3, "<A>")
        return vocab

    def build_char_vocab(self, words):
        counter = Counter(c for w in words for c in w)
        print('All characters: {0}'.format(len(counter)))
        char_vocab = [c for c, cnt in counter.items() if cnt > 3]  # 保留发生频次大于3的词
        print('Occurrence > 3 characters: {0}'.format(len(char_vocab)))

        char_vocab.insert(0, "<PAD>")
        char_vocab.insert(1, "<UNK>")
        char_vocab.insert(2, "<STA>")
        char_vocab.insert(3, "<END>")
        return char_vocab

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

    def process(self, parsed_text):
        output = {'word': [],
                  'lemma': [],
                  'pos': [],
                  'pos_id': [],
                  'ent': [],
                  'ent_id': [],
                  'offsets': [],
                  'sentences': []}

        for token in parsed_text:
            # [(token.text,token.idx) for token in parsed_sentence]
            output['word'].append(self._str(token.text))
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

    '''
     offsets based on raw_text
     this will solve the problem that, in raw_text, it's "a-b", in parsed test, it's "a - b"
    '''

    def get_raw_context_offsets(self, words, raw_text):
        raw_context_offsets = []
        p = 0
        for token in words:
            while p < len(raw_text) and re.match('\s', raw_text[p]):
                p += 1
            if raw_text[p:p + len(token)] != token:
                print('something is wrong! token', token, 'raw_text:', raw_text)

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

        def lower(text):
            return text.lower()

        return white_space_fix(remove_articles(remove_punc(lower(s))))

    # find the word id start and stop
    def find_span_with_gt(self, context, offsets, ground_truth):
        best_f1 = 0.0
        best_span = (len(offsets) - 1, len(offsets) - 1)
        gt = self.normalize_answer(pre_proc(ground_truth)).split()

        ls = [i for i in range(len(offsets)) if context[offsets[i][0]:offsets[i][1]].lower() in gt]

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

    def process_ans(self, ans):
        ans = ans.replace(" , ", ", ")
        ans = ans.replace(" . ", ". ")
        ans = ans.replace(" ? ", "? ")
        ans = ans.replace("? ^ ", "")

    # find the word id start and stop
    def find_span(self, offsets, start, end):
        start_index = -1
        end_index = -1
        for i, offset in enumerate(offsets):
            if (start_index < 0) or (start >= offset[0]):
                start_index = i
            if (end_index < 0) and (end <= offset[1]):
                end_index = i
        return (start_index, end_index)
