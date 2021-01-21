# -- coding:UTF-8 --

from datetime import datetime
import json
import numpy as np
import os
import random
import sys
import time
import torch
import collections
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from ranger.ranger import Ranger  # this is from ranger.py
from ranger.ranger import RangerVA  # this is from ranger913A.py
from ranger.ranger import RangerQH  # this is from rangerqh.py
from adamwr.adamw import AdamW
from adamwr.cyclic_scheduler import CyclicLRWithRestarts
from Models.Layers import MaxPooling, set_dropout_prob
from Utils.ConvQA_CNPreprocess import ConvQA_CNPreprocess
from Models.BaseTrainer import BaseTrainer
from Utils.CoQAUtils import AverageMeter, BatchGen, write_predictions,gen_upper_triangle, score
from Models.ConvQA_CN_Net import ConvQA_CN_Net
from Utils.CoQAPreprocess import CoQAPreprocess
from Utils.QuACPreprocess import QuACPreprocess
from evaluate import *
from Models.dataprocess import dataprocess

class ConvQA_CN_NetTrainer(BaseTrainer):
    def __init__(self, opt):
        super(ConvQA_CN_NetTrainer, self).__init__(opt)
        print('Model Trainer')
        set_dropout_prob(0.0 if not 'DROPOUT' in opt else float(opt['DROPOUT']))
        self.seed = int(opt['SEED'])
        self.opt = opt
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.opt['dataset'] == 'coqa':
            self.data_prefix = 'coqa-'
            self.preproc = CoQAPreprocess(self.opt)
        elif self.opt['dataset'] == 'quac':
            self.data_prefix = 'quac-'
            self.preproc = QuACPreprocess(self.opt)
        else:
            self.data_prefix = 'ConvQA_CN-'
            self.preproc = ConvQA_CNPreprocess(self.opt)
        if self.use_cuda:
            torch.cuda.manual_seed_all(self.seed)

        self.hidden = nn.Linear(self.opt['hidden_size']*2, 2, bias=True).cuda()

        ### seq2seq
        self.train_lang, self.dev_lang = dataprocess("train", "dev")
        self.opt['train_words']  = self.train_lang.n_words
        self.opt['dev_words'] = self.dev_lang.n_words


    def train(self):
        self.getSaveFolder()
        self.saveConf()
        self.result_file = self.opt['RESULT_FILE']
        self.vocab, self.vocab_embedding = self.preproc.load_data()
        self.log('-----------------------------------------------')
        self.log("Initializing model...")
        self.setup_model()
        batch_size = self.opt['BATCH_SIZE']
        if 'CHECK_POINT' in self.opt:
            model_path = os.path.join(self.opt['datadir'], self.opt['MODEL_PATH'])
            self.load_model(model_path)

        print('Loaing train json...')
        with open(os.path.join(self.opt['FEATURE_FOLDER'], self.data_prefix + 'train-preprocessed.json'), 'r') as f:
            train_data = json.load(f)

        with open(os.path.join(self.opt['FEATURE_FOLDER'], self.data_prefix + 'dev-preprocessed.json'), 'r') as f:
            dev_data = json.load(f)

        best_f1_score = 0
        num_epochs = self.opt['EPOCH']
        self.scheduler = CyclicLRWithRestarts(self.optimizer, batch_size, num_epochs, restart_period=5, t_mult=1.2,
                                         policy="cosine")
        for epoch in range(self.epoch_start, num_epochs):
            self.log('\n########Epoch {}########\n'.format(epoch))
            # self.network.train()
            start_time = datetime.now()
            train_batches = BatchGen(self.opt, train_data['data'], self.train_lang, self.use_cuda, Is_training=True)
            dev_batches = BatchGen(self.opt, dev_data['data'], self.dev_lang, self.use_cuda, Is_training=False)
            self.scheduler.step()
            ### step = 2700
            for i, batch in enumerate(train_batches):
                ''' 先判断是否进入测试阶段
                三个条件：
                    1.正常训练即将结束
                    2.训练刚开始，载入Check point
                    3.每1600步测试一次（参数可调）
                '''
                # if i == len(train_batches) - 1 or (epoch == 0 and i == 0 and ('CHECK_POINT' in self.opt)) or (i ==1800):
                # if (self.updates >= 0 and self.updates % 5000 == 0):
                if self.updates>0 and self.updates%1600==0:
                    print('Saving folder is', self.saveFolder)
                    print('Evaluating on dev set......')

                    #SDNET
                    # predictions = []
                    # confidence = []
                    # dev_answer = []
                    # final_json = []
                    # for j, dev_batch in enumerate(dev_batches):
                    #     phrase, phrase_score, pred_json, answers = self.predict(dev_batch)
                    #     final_json.extend(pred_json)
                    #     predictions.extend(phrase)
                    #     confidence.extend(phrase_score)
                    #     # dev_answer.extend(dev_batch[-3])  # answer_str
                    #     dev_answer.extend(answers)
                    # result, all_f1s = score(predictions, dev_answer, final_json)
                    # f1 = result['f1']
                    #
                    all_f1, final_json, all_loss = [], [], []

                    for j, dev_batch in enumerate(dev_batches):
                        loss, f1, pred_json, do_pre = self.predict(dev_batch)
                        if not do_pre:
                            continue
                        all_f1.append(f1)
                        all_loss.append(loss)
                        final_json.append(pred_json)

                    final_f1 = np.average(all_f1)
                    final_loss = np.average(all_loss)
                    print("Average F1 : {}".format(final_f1))
                    print("Best F1 : {}".format(max(max(all_f1))))
                    print("dev loss: ", final_loss)

                    if final_f1>best_f1_score:
                    # if f1 > best_f1_score:
                        model_file = os.path.join(self.result_file, 'best_model.pt')
                        self.save_for_predict(model_file, epoch)
                        best_f1_score = final_f1
                        pred_json_file = os.path.join(self.result_file, 'prediction.json')
                        with open(pred_json_file, 'a+', encoding='utf-8') as output_file:
                            json.dump(final_json, output_file, ensure_ascii=False)
                        # with open(pred_json_file, 'w', encoding='utf-8') as result_file:
                        #     json.dump("f1: {}".format(final_f1), result_file, ensure_ascii=False)
                        score_per_instance = []

                        ### 可以确定len(all_f1) = len(final_json)
                        for instance, s in zip(final_json, all_f1):
                            score_per_instance.append({
                                'id': instance[0]['id'],
                                'turn_id': instance[0]['turn_id'],
                                'f1': s[0]})

                        score_per_instance_json_file = os.path.join(self.result_file, 'score_per_instance.json')
                        with open(score_per_instance_json_file, 'w') as output_file:
                            json.dump(score_per_instance, output_file)

                    self.log("Epoch {0} - dev: F1: {1:.3f} (best F1: {2:.3f})\n".format(epoch, final_f1, best_f1_score))
                    # self.log("Results breakdown\n{0}".format(result))

                self.update(batch)
                if i % 100 == 0:
                    self.log('EPOCH[{0:2}] i[{1:4}] updates[{2:6}] train loss[{3:.5f}] remaining[{4}]'.format(
                        epoch, i, self.updates, self.train_loss.avg,
                        str((datetime.now() - start_time) / (i + 1) * (len(train_batches) - i - 1)).split('.')[0]))

            print("PROGRESS: {0:.2f}%".format(100.0 * (epoch + 1) / num_epochs))
            print('Config file is at ' + self.opt['confFile'])

    def setup_model(self):
        self.train_loss = AverageMeter()
        self.network = ConvQA_CN_Net(self.opt, self.train_lang, self.dev_lang)
        if self.use_cuda:
            self.log('Using GPU to setup model...')
            self.network.cuda()
        parameters = [p for p in self.network.parameters() if p.requires_grad]

        ## Ranger优化器
        self.optimizer = Ranger(parameters)
        # self.optimizer = AdamW(parameters, lr=1e-4, weight_decay=0.01)
        self.updates = 0
        self.epoch_start = 0
        self.loss_func = F.cross_entropy

    def update(self, batch):
        ### Train mode
        # if self.opt['dataset'] == 'coqa':
        #     gold_file = self.opt['CoQA_TRAIN_FILE']
        #     evaluate = CoQAEvaluator(self.opt, gold_file)
        # elif self.opt['dataset'] == 'quac':
        #     gold_file = self.opt['Quac_TRAIN_FILE']
        #     evaluate = CoQAEvaluator(self.opt, gold_file)
        # else:
        #     gold_file = self.opt['TRAIN_FILE']
        #     evaluate = CoQAEvaluator(self.opt, gold_file)
        self.network.train()
        self.network.drop_emb = True

        use_his = True
        x, x_offsets, x_bert_mask, q, q_mask, ground_truth, context_str, \
        context_words, context_word_offsets, span_answer_strs, is_max_context, answer_types, input_answer_strs, context_ids, turn_ids, his_inf_list = batch

        truth = []
        for i in range(len(ground_truth)):
            tmp = torch.LongTensor(ground_truth[i])
            tmp = torch.unsqueeze(tmp, 0)
            truth.append(tmp)
        ground_truth = torch.cat(truth)

        ### forward
        loss_list = self.network(x, x_bert_mask, q, q_mask, his_inf_list, input_answer_strs, ground_truth, context_str, context_ids, turn_ids, answer_types, is_max_context, True)
        # print("loss_list: ", loss_list)
        # start_logits, end_logits = self.network(x, x_bert_mask, q, q_mask, his_inf_list, input_answer_strs, self.updates, context_str, True)
        '''
        ###SDNET
        score_s, score_e, score_yes, score_no, score_no_answer = self.network(x, x_bert_mask, q_bert, his_inf_list, True)

        max_len = score_s.size(1)
        context_len = score_s.size(1)
        expand_score = gen_upper_triangle(score_s, score_e, max_len, self.use_cuda)

        scores = torch.cat((expand_score, score_no, score_yes, score_no_answer),
                           dim=1)  # batch x (context_len * context_len + 3)
        targets = []
        span_idx = int(context_len * context_len)
        for i in range(ground_truth.shape[0]):
            if ground_truth[i][0] == -1 and ground_truth[i][1] == -1:  # no answer
                targets.append(span_idx + 2)
            if ground_truth[i][0] == 0 and ground_truth[i][1] == -1:  # no
                targets.append(span_idx)
            if ground_truth[i][0] == -1 and ground_truth[i][1] == 0:  # yes
                targets.append(span_idx + 1)
            if ground_truth[i][0] != -1 and ground_truth[i][1] != -1:  # normal span
                targets.append(ground_truth[i][0] * context_len + ground_truth[i][1])

        targets = torch.LongTensor(np.array(targets))
        if self.use_cuda:
            targets = targets.cuda()
        # loss = self.loss_func(scores, targets)/(context_len*context_len)
        loss = self.loss_func(scores, targets)

        self.train_loss.update(loss.item(), 1)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.network.parameters(), self.opt['grad_clipping'])
        self.optimizer.step()
        self.updates += 1
        '''

        loss = sum(loss_list)/len(loss_list)
        # flood_value = torch.tensor([2.])
        # flood = torch.abs(loss - flood_value) + flood_value

        self.train_loss.update(loss.item(), 1)
        self.optimizer.zero_grad()
        loss.backward()
        # for name, p in self.network.named_parameters():
        #     a = p.grad
        #     if a == True:
        #         print(name, a)
        # for name, param in self.network.named_parameters():
        #     a = param.view(1,-1).squeeze(0).tolist()
        #     if sum(a)==0:
        #         print(param.size())
        #         print(name)
        #         print(param)
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.opt['grad_clipping'])
        self.optimizer.step()
        # self.scheduler.batch_step()
        self.updates += 1

    def predict(self, batch):
        self.network.eval()
        self.network.drop_emb = False

        # if self.opt['dataset'] == 'coqa':
        #     gold_file = self.opt['CoQA_DEV_FILE']
        #     evaluate = CoQAEvaluator(self.opt, gold_file)
        # elif self.opt['dataset'] == 'chinese':
        #     gold_file = self.opt['DEV_FILE']
        #     evaluate = CoQAEvaluator(self.opt, gold_file)
        # Run forward
        x, x_offsets, x_bert_mask, q, q_mask, ground_truth, context_str, \
        context_words, context_word_offsets, span_answer_strs, is_max_context, answer_types, input_answer_strs, context_ids, turn_ids, his_inf_list = batch

        loss, f1_list, pred_json, do_pre = self.network(x, x_bert_mask, q, q_mask, his_inf_list, input_answer_strs, ground_truth, context_str, context_ids, turn_ids, answer_types, is_max_context, False)
        return loss, f1_list, pred_json, do_pre
        # '''
        # score_s, score_e, score_yes, score_no, score_no_answer = self.network(x, x_bert_mask, q_bert, his_inf_list, True)
        # # truth = []
        # # for i in range(len(ground_truth)):
        # #     if ground_truth[i][0] == None: continue
        # #
        # #     tmp = torch.LongTensor(ground_truth[i])
        # #     tmp = torch.unsqueeze(tmp, 0)
        # #     truth.append(tmp)
        # # ground_truth = torch.cat(truth)
        #
        # batch_size = score_s.shape[0]
        # max_len = score_s.size(1)
        # context_len = score_s.size(1)
        # expand_score = gen_upper_triangle(score_s, score_e, max_len, self.use_cuda)
        # scores = torch.cat((expand_score, score_no, score_yes, score_no_answer),
        #                    dim=1)  # batch x (context_len * context_len + 3)
        # prob = F.softmax(scores, dim=1).data.cpu()  # Transfer to CPU/normal tensors for numpy ops
        #
        # # Get argmax text spans
        # predictions = []
        # confidence = []
        # pred_json = []
        # answers = []
        # for i in range(batch_size):
        #     _, ids = torch.sort(prob[i, :], descending=True)
        #     idx = 0
        #     best_id = ids[idx]
        #
        #     if best_id < context_len * context_len:
        #         st = best_id / context_len
        #         ed = best_id % context_len
        #         if ed>=len(context_word_offsets[i]):
        #             continue
        #         # print(st, ed)
        #         # print(len(context_word_offsets[i]))
        #         st = context_word_offsets[i][st][0]
        #         ed = context_word_offsets[i][ed][1]
        #         predictions.append(context_str[i][st:ed])
        #
        #     if best_id == context_len * context_len:
        #         predictions.append('no')
        #
        #     if best_id == context_len * context_len + 1:
        #         predictions.append('yes')
        #
        #     if best_id == context_len * context_len + 2:
        #         predictions.append('unknown')
        #     confidence.append(float(prob[i, best_id]))
        #     pred_json.append({
        #         'id': context_ids[i],
        #         'turn_id': turn_ids[i],
        #         'answer': predictions[-1]
        #     })
        #     answers.append(answer_strs[i])
        #
        # return (predictions, confidence, pred_json, answers)  # list of strings, list of floats, list of jsons
        # '''
        # _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        #     "PrelimPrediction",
        #     ["start_index", "end_index", "start_logit", "end_logit"])
        # batch_size = start_logits.size()[0]
        # seq_length = x[0].size()[1]
        # loss_list = []
        # f1_list = []
        # pred_json = []
        # do_pre = True#用于判断是否有feature
        # prelim_predictions = []
        #
        # for i in range(batch_size):
        #     ### start, end 都是None
        #     start_positions = ground_truth[i][0]
        #     end_positions = ground_truth[i][1]
        #
        #     if start_positions==None:
        #         continue
        #         ### ans = "无法回答"
        #         ### 有可能会跳过所有判断
        #     if start_positions>=seq_length or end_positions>=seq_length:
        #         continue
        #
        #     start_loss, _ = self.compute_loss(start_logits[i], seq_length, start_positions)
        #     end_loss, _ = self.compute_loss(end_logits[i], seq_length, end_positions)
        #
        #     s_prob = F.softmax(start_logits[i], dim=-1)
        #     e_prob = F.softmax(end_logits[i], dim=-1)
        #     s, e = s_prob.tolist(), e_prob.tolist()
        #     sp, ep = s.index(max(s)), e.index(max(e))
        #     start_indexes = self._get_best_indexes(start_logits[i], n_best_size)
        #     end_indexes = self._get_best_indexes(end_logits[i], n_best_size)
        #     for start_index in start_indexes:
        #         for end_index in end_indexes:
        #             # We could hypothetically create invalid predictions, e.g., predict
        #             # that the start of the span is in the question. We throw out all
        #             # invalid predictions.
        #             if start_index >= seq_length:
        #                 continue
        #             if end_index >= seq_length:
        #                 continue
        #             if end_index < start_index:
        #                 continue
        #             length = end_index - start_index + 1
        #             if length > self.opt['ans_max_len']:
        #                 continue
        #             prelim_predictions.append(
        #                 _PrelimPrediction(
        #                     start_index=start_index,
        #                     end_index=end_index,
        #                     start_logit=start_logits[i][start_index],
        #                     end_logit=end_logits[i][end_index],
        #                 ))
        #     if prelim_predictions:
        #         prelim_predictions = sorted(
        #             prelim_predictions,
        #             key=lambda x: (x.start_logit + x.end_logit),
        #             reverse=True)
        #         _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        #             "NbestPrediction", ["text", "start_logit", "end_logit"])
        #         nbest = []
        #         pred = prelim_predictions[0]
        #         if len(nbest) >= n_best_size:
        #             continue
        #         if self.opt['dataset'] == 'chinese':
        #             context = "".join(context_str[i])
        #             tok_tokens = context[pred.start_index:(pred.end_index + 1)]
        #             tok_text = tok_tokens
        #         else:
        #             tok_tokens = context_str[i][pred.start_index:(pred.end_index + 1)]
        #             tok_text = " ".join(tok_tokens)
        #
        #         orig_tokens = span_answer_strs[i]
        #
        #         if self.opt['dataset'] == ("coqa" or 'quac'):
        #             tok_text = tok_text.replace(" ##", "")
        #             tok_text = tok_text.replace("##", "")
        #         tok_text = tok_text.strip()
        #         tok_text = " ".join(tok_text.split())
        #         orig_text = orig_tokens
        #
        #         if self.opt['dataset'] == 'quac' or self.opt['dataset'] == 'coqa':
        #             f1 = self.f1_score(tok_text, orig_text)
        #         else:
        #             f1 = evaluate.compute_f1(orig_text, tok_text, self.opt['dataset'])
        #             # f1 = self.f1_score(tok_text, orig_text)
        #         # if f1>0:
        #         #     print("tok_text: ", tok_text)
        #         #     print("orig_text: ", orig_text)
        #         #     print("f1: ", f1)
        #         # else:
        #         #     f1 = evaluate.compute_f1(ans, context_str[i][sp:ep])
        #         loss = (start_loss + end_loss) / 2.0
        #         loss_list.append(loss.unsqueeze(0))
        #         f1_list.append(f1)
        #         pred_json.append({
        #             'id': context_ids[i],
        #             'turn_id': turn_ids[i],
        #             'answer': orig_text,
        #             'predict': tok_text
        #         })
        #     '''
        #     ans = answer_strs[i]
        #     if sp > ep:
        #         continue
        #     if self.opt['dataset'] == 'coqa':
        #         # print("pre: ", " ".join(context_str[i][sp:ep]))
        #         # print("pos: ", " ".join(context_str[i][start_positions: end_positions]))
        #         # print("ans:", "".join(ans))
        #         if type(answer_strs[i]) == str:
        #             f1 = evaluate.compute_f1(ans, self.process_ans(" ".join(context_str[i][sp:ep])))
        #         else:
        #             f1 = evaluate.compute_f1("".join(ans), self.process_ans(" ".join(context_str[i][sp:ep])))
        #     else:
        #         f1 = evaluate.compute_f1(ans, context_str[i][sp:ep])
        #     '''
        #
        #
        # # print("f1: ", f1_list,len(f1_list))
        # # print("loss: ", loss_list, len(loss_list))
        # if loss_list:
        #
        #     ### 最后一轮的batch_size较小，给最后一轮补齐
        #     avg_f1 = np.average((f1_list))
        #     while len(f1_list)<list_len:
        #         f1_list.append(avg_f1)
        #
        #     loss = torch.cat(loss_list, 0)
        #     loss = torch.mean(loss, dim=0)
        #     loss = round(loss.item(), 4)
        #
        #     return loss, f1_list, pred_json, do_pre
        # else:
        #     do_pre = False
        #     return [], [], [], do_pre

    def load_model(self, model_path):
        print('Loading model from', model_path)
        checkpoint = torch.load(model_path)
        state_dict = checkpoint['state_dict']
        new_state = set(self.network.state_dict().keys())
        for k in list(state_dict['network'].keys()):
            if k not in new_state:
                del state_dict['network'][k]
        for k, v in list(self.network.state_dict().items()):
            if k not in state_dict['network']:
                state_dict['network'][k] = v
        self.network.load_state_dict(state_dict['network'])

        print('Loading finished', model_path)

    def save(self, filename, epoch, prev_filename):
        params = {
            'state_dict': {
                'network': self.network.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'updates': self.updates  # how many updates
            },
            'train_loss': {
                'val': self.train_loss.val,
                'avg': self.train_loss.avg,
                'sum': self.train_loss.sum,
                'count': self.train_loss.count
            },
            'config': self.opt,
            'epoch': epoch
        }
        try:
            torch.save(params, filename)
            self.log('model saved to {}'.format(filename))
            if os.path.exists(prev_filename):
                os.remove(prev_filename)
        except BaseException:
            self.log('[ WARN: Saving failed... continuing anyway. ]')

    def save_for_predict(self, filename, epoch):
        network_state = dict([(k, v) for k, v in self.network.state_dict().items() if
                              k[0:4] != 'CoVe' and k[0:4] != 'ELMo' and k[0:9] != 'AllenELMo' and k[0:4] != 'Bert'])

        if 'eval_embed.weight' in network_state:
            del network_state['eval_embed.weight']
        if 'fixed_embedding' in network_state:
            del network_state['fixed_embedding']
        params = {
            'state_dict': {'network': network_state},
            'config': self.opt,
        }
        try:
            torch.save(params, filename)
            self.log('model saved to {}'.format(filename))
        except BaseException:
            self.log('[ WARN: Saving failed... continuing anyway. ]')

    def process_ans(self, ans):
        ans = ans.replace(" , ", ", ")
        ans = ans.replace(" . ", ". ")
        ans = ans.replace(" ? ", "? ")
        ans = ans.replace("^ ", "")
        ans = ans.replace(" ^ ", "")
        ans = ans.replace("? ^ ", "")
        return ans

    def _get_best_indexes(self, logits, n_best_size):
        """Get the n-best logits from a list."""
        index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

        best_indexes = []
        for i in range(len(index_and_score)):
            if i >= n_best_size:
                break
            best_indexes.append(index_and_score[i][0])
        return best_indexes

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

