import sys
import os
import json
from io import open
import unicodedata
import string
import re
import random
import pdb
SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r"", s)
    s = re.sub(r"[^a-zA-Z.!?]", r" ", s)
    return s

def readLangs(train_data, dev_data):
    print("Reading {} data for seq2seq model...".format(train_data))
    print("Reading {} data for seq2seq model...".format(dev_data))

    # Read the file and split into lines
    # train_path = 'data/quac_original/' + train_data +'_v0.2.json'
    # dev_path = 'data/quac_original/' + 'val' +'_v0.2.json'
    train_path = 'data/quac_original/train_small.json'
    dev_path = 'data/quac_original/val_small.json'

    with open(train_path, 'r') as f1:
        train_dataset = json.load(f1)
    with open(dev_path, 'r') as f2:
        dev_dataset = json.load(f2)

    ### train lang包含train数据集所有信息，dev lang包含dev数据集除答案外所有信息
    train_text = ""
    dev_text = ""
    # for i, data in enumerate(train_dataset['data']):
    #     story = data['story']
    #     story = normalizeString(story)# 处理文本，取出标点等
    #     train_text += " " + story
    #     ques = data['questions']
    #     for q in ques:
    #         Q = normalizeString(q["input_text"])
    #         train_text += " " + Q
    #     ans = data['answers']
    #     for a in ans:
    #         A = normalizeString(a["input_text"])
    #         dev_text += " " + A
    #         train_text += " " + A
    #
    # for i, data in enumerate(dev_dataset['data']):
    #     story = data['story']
    #     story = normalizeString(story)# 处理文本，取出标点等
    #     train_text += " " + story
    #     ques = data['questions']
    #     for q in ques:
    #         Q = normalizeString(q["input_text"])
    #         train_text += " " + Q
    #     ans = data['answers']
    #     for a in ans:
    #         A = normalizeString(a["input_text"])
    #         dev_text += " " + A
    #         train_text += " " + A
    for dataset in [train_dataset['data'], dev_dataset['data']]:
        for i, data in enumerate(dataset):
            feature = data['paragraphs'][0]
            story = feature['context']
            story = normalizeString(story)# 处理文本，取出标点等
            train_text += " " + story
            dev_text += " " + story
            qas = feature['qas']
            for qa in qas:
                ques = qa['question']
                ans = qa['answers'][0]['text']
                Q = normalizeString(ques)
                train_text += " " + Q
                dev_text += " " + Q
                A = normalizeString(ans)
                dev_text += " " + A
                train_text += " " + A

    train_lang = Lang(train_data)
    train_lang.addSentence((train_text))
    dev_lang = Lang(dev_data)
    dev_lang.addSentence((dev_text))

    return train_lang, dev_lang


eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def prepareData(train, dev):
    train_lang, dev_lang = readLangs(train, dev)
    print("Counted words:")
    print(dev_lang.name, dev_lang.n_words)
    return train_lang, dev_lang

def dataprocess(train, dev):
    train_lang, dev_lang = prepareData(train, dev)
    return train_lang, dev_lang

