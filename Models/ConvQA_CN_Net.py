# -- coding:UTF-8 --

import math
import random
import numpy as np
from numpy import inf
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
from transformers.tokenization_bert import BasicTokenizer
from Models.Layers import MaxPooling, CNN, dropout, RNN_from_opt, set_dropout_prob, weighted_avg, set_seq_dropout, Attention, DeepAttention, LinearSelfAttn, GetFinalScores
from Utils.CoQAUtils import POS, ENT, tensorFromSentence
from Models.dataprocess import dataprocess, normalizeString
from rouge import *
from model.modeling_auto import MODEL_FOR_CONVERSATIONAL_QUESTION_ANSWERING_MAPPING, AutoModelForConversationalQuestionAnswering
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
def to_list(tensor):
    return tensor.detach().cpu().tolist()


class ConvQA_CN_Net(nn.Module):
    def __init__(self, opt, dev_l):
        super(ConvQA_CN_Net, self).__init__()
        print('ConvQA_CN_Net model\n')
        self.opt = opt
        self.use_history = self.opt['use_history']
        self.dev_lang = dev_l

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
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.pretrain_path,
                do_lower_case=True,
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
        self.max_answer_length = self.opt['max_answer_length']
        self.loss_ratio = self.opt['loss_ratio']
        self.max_length = self.opt['max_featrue_length']

        self.criterion = nn.NLLLoss()
        self.num_layers = 2

        self.generate_embedding = nn.Embedding(self.opt['dev_words'], self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.opt['dev_words'])

    def forward(self, x, x_bert_mask, rational_mask, x_segment, q, q_mask, his_info_list, answer_strs, ex_pre, ground_truth, context_str, context_ids, turn_ids, answer_types, cls_idx, is_max_context, token_to_orig_map, followup, yesno, is_training=False):
        if is_training:
            self.batch_size = self.opt['TRAIN_BATCH_SIZE']
        else:
            self.batch_size = self.opt['DEV_BATCH_SIZE']
        x_bert = torch.cat(x, 0)
        x_mask = torch.cat(x_bert_mask, 0)
        x_sep = torch.cat(x_segment, 0)
        follow = torch.cat(followup, 0)
        yesno = torch.cat(yesno, 0)
        rational_mask = torch.cat(rational_mask, 0)
        cls_idx = torch.LongTensor(cls_idx).cuda()

        if is_training:
            history, history_mask, history_sep = None, None, None
            if self.use_history:
                new_his, new_his_mask, new_his_seg = [], [], []
                for his in his_info_list:
                    new_his += his['his']
                    new_his_mask += his['his_mask']
                    new_his_seg += his['his_seg']

                history = torch.cat(new_his, 0).cuda()
                history_mask = torch.cat(new_his_mask, 0).cuda()
                history_sep = torch.cat(new_his_seg, 0).cuda()

            start, end = [], []
            for i in range(x_bert.size(0)):
                start.append(ground_truth[i][0].unsqueeze(0))
                end.append(ground_truth[i][1].unsqueeze(0))
            start = torch.cat(start, 0).cuda()
            end = torch.cat(end, 0).cuda()
            inputs = {
                "input_ids": x_bert,
                "token_type_ids": x_sep,
                "attention_mask": x_mask,
                "start_positions": start,
                "end_positions": end,
                "rational_mask": rational_mask,
                "cls_idx": cls_idx,
                "history": history,
                "history_mask": history_mask,
                "history_sep": history_sep,
                "use_history": self.use_history,
                "answer_types": answer_types,
                "follow": follow,
                "yesno": yesno,
                "is_training": is_training,
            }
            ex_loss, ge_tensor, ge_id = self.model(**inputs)
            batch_size = x_bert.size(0)
            if ge_id:
                ge_loss = self.trainIters(ge_tensor, answer_strs, ge_id)
                ex_ratio = (batch_size - len(ge_id))/batch_size
                ge_ratio = len(ge_id)/batch_size
                loss = (ex_loss * ex_ratio + ge_loss * ge_ratio) * (1 - self.loss_ratio)
            else:
                loss = ex_loss * (1 - self.loss_ratio)
                # loss = ex_loss
            # if loss<0 or loss>100:
            #     pdb.set_trace()
            return loss
        else:
            with torch.no_grad():
                # inputs = {
                #     "input_ids": x_bert,
                #     "token_type_ids": x_sep,
                #     "attention_mask": x_mask,
                #     "start_positions": start,
                #     "end_positions": end,
                #     "cls_idx": cls_idx,
                # }
                # loss = self.model(**inputs)

                inputs = {
                    "input_ids": x_bert,
                    "token_type_ids": x_sep,
                    "attention_mask": x_mask,
                }
                outputs, encoder_output, ge_id, ex_id = self.model(**inputs)
                if not ge_id:
                    pred_json, all_predictions, all_nbest_json = self.extract_evaluate(outputs, ex_pre,
                                                                                       context_str,
                                                                                       context_ids, turn_ids,
                                                                                       is_max_context,
                                                                                       token_to_orig_map, ex_id)

            return pred_json, all_predictions, all_nbest_json

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


        return loss_list

    def extract_evaluate(self, outputs, answer_strs, context_str, context_ids, turn_ids, is_max_context, token_to_orig_map, ex_id):

        batch_size = len(answer_strs)
        _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "PrelimPrediction",
            ["start_index", "end_index", "score", "cls_idx"])
        n_best_size = self.opt['N_BEST']
        pred_json = []
        all_predictions = []
        all_nbest_json = collections.OrderedDict()
        prelim_predictions = []
        CLS_YES = 0
        CLS_NO = 1
        CLS_UNK = 2
        CLS_SPAN = 3

        for k, i in enumerate(ex_id):
        # for i in range(batch_size):

            output = [to_list(output[k]) for output in outputs]
            # output = [to_list(output[i]) for output in outputs]
            score_yes, score_no, score_span, score_unk = -float('INF'), -float('INF'), -float('INF'), float('INF')
            start_logits, end_logits, yes_logits, no_logits, unk_logits = output
            feature_yes_score, feature_no_score, feature_unk_score = \
                yes_logits[0] * 2, no_logits[0] * 2, unk_logits[0] * 2
            start_indexes, end_indexes = self._get_best_indexes(start_logits, n_best_size), \
                                         self._get_best_indexes(end_logits, n_best_size)
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
                    feature_span_score = start_logits[start_index] + end_logits[end_index]
                    prelim_predictions.append(
                        _PrelimPrediction(
                            start_index=start_index,
                            end_index=end_index,
                            score=feature_span_score,
                            cls_idx=CLS_SPAN,
                        ))

            if feature_unk_score < score_unk:  # find min score_noanswer
                score_unk = feature_unk_score
            if feature_yes_score > score_yes:  # find max score_yes
                score_yes = feature_yes_score
            if feature_no_score > score_no:  # find max score_no
                score_no = feature_no_score
            prelim_predictions.append(
                _PrelimPrediction(start_index=0,
                                  end_index=0,
                                  score=score_unk,
                                  cls_idx=CLS_UNK))
            prelim_predictions.append(
                _PrelimPrediction(start_index=0,
                                  end_index=0,
                                  score=score_yes,
                                  cls_idx=CLS_YES))
            prelim_predictions.append(
                _PrelimPrediction(start_index=0,
                                  end_index=0,
                                  score=score_no,
                                  cls_idx=CLS_NO))

            prelim_predictions = sorted(prelim_predictions,
                                        key=lambda p: p.score,
                                        reverse=True)

            _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
                "NbestPrediction", ["text", "answer", "score", "cls_idx", "type"]
            )
            nbest = []
            seen_predictions = {}
            for pred in prelim_predictions:
                if len(nbest) >= n_best_size:
                    break

                orig_tokens = answer_strs[i]
                orig_text = self.tokenizer.convert_tokens_to_string(orig_tokens)
                orig_text = orig_text.strip()
                orig_text = " ".join(orig_text.split())

                if pred.cls_idx == CLS_SPAN:
                    tok_tokens = context_str[i][pred.start_index:(pred.end_index + 1)]
                    tok_text = self.tokenizer.convert_tokens_to_string(tok_tokens)
                    tok_text = tok_text.strip()
                    tok_text = " ".join(tok_text.split())
                    final_text = self.get_final_text(tok_text, orig_text, True)
                    if final_text in seen_predictions:
                        continue
                    seen_predictions[final_text] = True
                    nbest.append(_NbestPrediction(text=final_text, answer=orig_text, score=pred.score, cls_idx=pred.cls_idx, type='extractive'))
                else:
                    text = ['yes', 'no', 'unknown']
                    nbest.append(_NbestPrediction(text=text[pred.cls_idx], answer=orig_text, score=pred.score, cls_idx=pred.cls_idx, type='extractive'))
            if len(nbest) < 1:
                nbest.append(_NbestPrediction(text='unknown', answer=orig_text, score=-float('inf'), cls_idx=CLS_UNK))
            assert len(nbest) >= 1

            probs = self._compute_softmax([p.score for p in nbest])
            nbest_json = []

            for index, entry in enumerate(nbest):
                pred_json.append({
                    'id': context_ids[i],
                    'turn_id': turn_ids[i],
                    'answer': entry.answer,
                    'predict': entry.text,
                    'type': entry.type
                })

                output = collections.OrderedDict()
                output["text"] = entry.text
                output["probability"] = probs[index]
                output["score"] = entry.score
                nbest_json.append(output)
            assert len(nbest_json) >= 1
            _id = turn_ids[i]
            _turn_id = _id[:-4]
            all_predictions.append({
                'id': _turn_id,
                'turn_id': _id,
                'answer': self.confirm_preds(nbest_json)
            })
            qas_id = _id
            all_nbest_json[qas_id] = nbest_json

        return pred_json, all_predictions, all_nbest_json

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

    def generative_loss(self, batch_size, target_tensor, encoder_output):
        #
        # seq_len = encoder_output.size(1)
        # hidden = encoder_output.view(seq_len, batch_size, -1)
        # decoder_hidden = hidden[0].view(batch_size, 1, -1)
        decoder_hidden = torch.zeros(batch_size, 1, self.hidden_size).cuda()

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

    def trainIters(self, encoder_output, answer_strs, ge_id):
        target = []

        for i in ge_id:
            answers = normalizeString(answer_strs[i])
            tmp, _ = tensorFromSentence(self.dev_lang, answers, len(answers))
            tmp = tmp.view(-1, 1)
            target.append(tmp)
        batch_size = encoder_output.size(0)


        # print("batch_size: ", batch_size)
        # print("target_tensor: ", target_tensor.size())
        # print("encoder_output: ", encoder_output.size())

        loss = self.generative_loss(batch_size, target, encoder_output)
        return sum(loss)/len(loss)

    def generate_predict(self, encoder_output):
        # with torch.no_grad():
        EOS_token = 1
        SOS_token = 0
        max_length = self.max_answer_length

        batch_size = encoder_output.size(0)

        # decoder_input = torch.ones([batch_size, 1], dtype=torch.long).cuda()# SOS 1, 1

        decoder_hidden = torch.zeros(batch_size, 1, self.hidden_size).cuda()

        words_list = []

        # decoder_attentions = torch.zeros(max_length, max_length)
        for i in range(batch_size):
            decoded_words = []
            hidden = decoder_hidden[i].cuda()
            decoder_input = torch.tensor([[SOS_token]], dtype=torch.long).cuda()
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
                decoder_input = topi.squeeze().detach()
            words_list.append(decoded_words)

        return words_list

    def generate_evaluate(self, ge_id, predict, answers, context_ids, turn_ids):

        pred_json = []
        all_predictions = []
        all_nbest_json = collections.OrderedDict()

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "answer", "type"]
        )
        nbest = []
        for k, i in enumerate(ge_id):
            tok_text = self.tokenizer.convert_tokens_to_string(predict[k])
            tok_text = tok_text.strip()
            tok_text = " ".join(tok_text.split())
            final_text = self.get_final_text(tok_text, " ".join(answers[i]), True)
            nbest.append(
                _NbestPrediction(text=final_text, answer=" ".join(answers[i]), type='generative'))

        nbest_json = []
        for index, entry in enumerate(nbest):
            pred_json.append({
                'id': context_ids[ge_id[index]],
                'turn_id': turn_ids[ge_id[index]],
                'answer': entry.answer,
                'predict': entry.text,
                'type': entry.type
            })
            output = collections.OrderedDict()
            output["text"] = entry.text
            nbest_json.append(output)

            _id, _turn_id = context_ids[ge_id[index]], turn_ids[ge_id[index]]
            all_predictions.append({
                'id': _id,
                'turn_id': int(_turn_id),
                'answer': self.confirm_preds(nbest_json)
            })
            qas_id = _id + ' ' + str(_turn_id)
            all_nbest_json[qas_id] = nbest_json

        return pred_json, all_predictions, all_nbest_json

    def confirm_preds(self, nbest_json):
        # Do something for some obvious wrong-predictions
        # TODO: can do more things?
        subs = [
            'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
            'ten', 'eleven', 'twelve', 'true', 'false'
        ]
        ori = nbest_json[0]['text']
        if len(ori) < 2:  # mean span like '.', '!'
            for e in nbest_json[1:]:
                if self._normalize_answer(e['text']) in subs:
                    return e['text']
            return 'unknown'
        return ori

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

    def _normalize_answer(self, s):
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

    def _compute_softmax(self, scores):
        """Compute softmax probability over raw logits."""
        if not scores:
            return []

        max_score = None
        for score in scores:
            if max_score is None or score > max_score:
                max_score = score

        exp_scores = []
        total_sum = 0.0
        for score in scores:
            x = math.exp(score - max_score)
            exp_scores.append(x)
            total_sum += x

        probs = []
        for score in exp_scores:
            probs.append(score / total_sum)
        return probs

    def get_final_text(self, pred_text, orig_text, do_lower_case, verbose_logging=False):
        """Project the tokenized prediction back to the original text."""

        # When we created the data, we kept track of the alignment between original
        # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
        # now `orig_text` contains the span of our original text corresponding to the
        # span that we predicted.
        #
        # However, `orig_text` may contain extra characters that we don't want in
        # our prediction.
        #
        # For example, let's say:
        #   pred_text = steve smith
        #   orig_text = Steve Smith's
        #
        # We don't want to return `orig_text` because it contains the extra "'s".
        #
        # We don't want to return `pred_text` because it's already been normalized
        # (the SQuAD eval script also does punctuation stripping/lower casing but
        # our tokenizer does additional normalization like stripping accent
        # characters).
        #
        # What we really want to return is "Steve Smith".
        #
        # Therefore, we have to apply a semi-complicated alignment heuristic between
        # `pred_text` and `orig_text` to get a character-to-character alignment. This
        # can fail in certain cases in which case we just return `orig_text`.

        def _strip_spaces(text):
            ns_chars = []
            ns_to_s_map = collections.OrderedDict()
            for (i, c) in enumerate(text):
                if c == " ":
                    continue
                ns_to_s_map[len(ns_chars)] = i
                ns_chars.append(c)
            ns_text = "".join(ns_chars)
            return (ns_text, ns_to_s_map)

        # We first tokenize `orig_text`, strip whitespace from the result
        # and `pred_text`, and check if they are the same length. If they are
        # NOT the same length, the heuristic has failed. If they are the same
        # length, we assume the characters are one-to-one aligned.
        tokenizer = BasicTokenizer(do_lower_case=True)
        tok_text = " ".join(tokenizer.tokenize(orig_text))

        start_position = tok_text.find(pred_text)
        if start_position == -1:
            if verbose_logging:
                logger.info("Unable to find text: '%s' in '%s'" %
                            (pred_text, orig_text))
            return orig_text
        end_position = start_position + len(pred_text) - 1

        (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
        (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

        if len(orig_ns_text) != len(tok_ns_text):
            if verbose_logging:
                logger.info("Length not equal after stripping spaces: '%s' vs '%s'", orig_ns_text, tok_ns_text)
            return orig_text

        # We then project the characters in `pred_text` back to `orig_text` using
        # the character-to-character alignment.
        tok_s_to_ns_map = {}
        for (i, tok_index) in tok_ns_to_s_map.items():
            tok_s_to_ns_map[tok_index] = i

        orig_start_position = None
        if start_position in tok_s_to_ns_map:
            ns_start_position = tok_s_to_ns_map[start_position]
            if ns_start_position in orig_ns_to_s_map:
                orig_start_position = orig_ns_to_s_map[ns_start_position]

        if orig_start_position is None:
            if verbose_logging:
                logger.info("Couldn't map start position")
            return orig_text

        orig_end_position = None
        if end_position in tok_s_to_ns_map:
            ns_end_position = tok_s_to_ns_map[end_position]
            if ns_end_position in orig_ns_to_s_map:
                orig_end_position = orig_ns_to_s_map[ns_end_position]

        if orig_end_position is None:
            if verbose_logging:
                logger.info("Couldn't map end position")
            return orig_text

        output_text = orig_text[orig_start_position:(orig_end_position + 1)]
        return output_text