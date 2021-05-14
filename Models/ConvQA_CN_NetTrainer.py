# -- coding:UTF-8 --

from datetime import datetime
import json
import numpy as np
import os
import random
import sys
import time
import pdb
import torch
import collections
import logging
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from ranger.ranger import Ranger  # this is from ranger.py
from ranger.ranger import RangerVA  # this is from ranger913A.py
from ranger.ranger import RangerQH  # this is from rangerqh.py
# from adamwr.adamw import AdamW
# from adamwr.cyclic_scheduler import CyclicLRWithRestarts
from Models.Layers import MaxPooling, set_dropout_prob
from Utils.ConvQA_CNPreprocess import ConvQA_CNPreprocess
from Models.BaseTrainer import BaseTrainer
from Utils.CoQAUtils import AverageMeter, BatchGen, write_predictions,gen_upper_triangle, score
from Models.ConvQA_CN_Net import ConvQA_CN_Net
from Utils.QuACPreprocess import QuACPreprocess
from Models.dataprocess import dataprocess
from evaluate import eval_fn
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)
fileHandler = logging.FileHandler('{}.log'.format(datetime.now().strftime("%Y-%m-%d-%H:%M:%S")))
logger.addHandler(fileHandler)

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
        if self.opt['dataset'] == 'quac':
            self.data_prefix = 'quac-'
            self.preproc = QuACPreprocess(self.opt)
        if self.use_cuda:
            torch.cuda.manual_seed_all(self.seed)

        ### seq2seq
        self.train_lang, self.dev_lang = dataprocess("train", "dev")
        self.opt['train_words'] = self.train_lang.n_words
        self.opt['dev_words'] = self.dev_lang.n_words

    def train(self):
        self.getSaveFolder()
        self.saveConf()
        self.result_file = self.opt['RESULT_FILE']
        self.log('-----------------------------------------------')
        self.log("Initializing model...")
        self.setup_model()

        if 'CHECK_POINT' in self.opt:
            model_path = os.path.join(self.opt['datadir'], self.opt['CHECK_POINT_PATH'])
            self.load_model(model_path)

        print('Loaing train json...')
        with open(os.path.join(self.opt['FEATURE_FOLDER'], self.data_prefix + 'train-preprocessed.json'), 'r') as f:
            train_data = json.load(f)

        with open(os.path.join(self.opt['FEATURE_FOLDER'], self.data_prefix + 'dev-preprocessed.json'), 'r') as f:
            dev_data = json.load(f)

        output_prediction_file = self.opt['OUTPUT_FILE'] + "prediction_file.json"
        best_f1_score = 0
        last_epoch = 0
        num_epochs = self.opt['EPOCH']
        # self.scheduler = CyclicLRWithRestarts(self.optimizer, batch_size, num_epochs, restart_period=5, t_mult=1.2,
        #                                  policy="cosine")
        for epoch in range(self.epoch_start, num_epochs):
            ### best_f1_score记录每个epoch里的最优值
            self.log('\n########Epoch {}########\n'.format(epoch))
            # self.network.train()
            start_time = datetime.now()
            train_batches = BatchGen(self.opt, train_data['data'], self.use_cuda, is_training=True)
            dev_batches = BatchGen(self.opt, dev_data['data'], self.use_cuda, is_training=False)
            # self.scheduler.step()
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
                # if self.updates>0 and self.updates%1000==0:
                if self.updates > 0 and self.updates % 1000 == 0:

                    print('Saving folder is', self.saveFolder)
                    print('Evaluating on dev set......')

                    final_json, all_predictions_list, all_nbest_json_list = [], [], []
                    results = {}
                    count = 0
                    for j, dev_batch in enumerate(dev_batches):
                        pred_json, all_predictions, all_nbest_json = self.predict(dev_batch)
                        count += len(pred_json)
                        final_json.append(pred_json)
                        all_predictions_list += all_predictions

                    with open(output_prediction_file, "w") as writer:
                        writer.write(json.dumps(all_predictions_list, indent=4) + "\n")
                    with open(self.opt['Quac_DEV_FILE'], 'r') as f:
                        val_file = json.load(f)
                    val_file = val_file['data']

                    new = {}
                    for r in all_predictions_list:
                        tmp = {r['turn_id']: [r['answer'], 'y', 'y']}
                        if r['id'] in new:
                            new[r['id']][r['turn_id']] = [r['answer'], 'y', 'y']
                        else:
                            new[r['id']] = {}
                            new[r['id']][r['turn_id']] = [r['answer'], 'y', 'y']
                            
                    metric_json = eval_fn(val_file, new, False)
                    # logger.info("Results: {}".format(results))
                    final_f1 = metric_json['f1']
                    # pdb.set_trace()
                    if best_f1_score != 0:
                        print("Best F1 : {}".format(max(final_f1, best_f1_score)))
                    # print("dev loss: ", final_loss)

                    if final_f1>best_f1_score:
                        model_file = os.path.join(self.result_file, 'best_model.pt')
                        self.save_for_predict(model_file, epoch)
                        best_f1_score = final_f1
                        pred_json_file = os.path.join(self.result_file, 'prediction.json')
                        with open(pred_json_file, 'w', encoding='utf-8') as output_file:
                            json.dump(final_json, output_file, ensure_ascii=False)
                        # with open(pred_json_file, 'w', encoding='utf-8') as result_file:
                        #     json.dump("f1: {}".format(final_f1), result_file, ensure_ascii=False)
                        score_per_instance = []

                        ### 可以确定len(all_f1) = len(final_json)
                        for instance in final_json:
                            score_per_instance.append({
                                'id': instance[0]['turn_id'],
                                'turn_id': instance[0]['id']})

                        score_per_instance_json_file = os.path.join(self.result_file, 'score_per_instance.json')
                        with open(score_per_instance_json_file, 'w') as output_file:
                            json.dump(score_per_instance, output_file)

                    self.log("Epoch {0} - dev: F1: {1:.3f} (best F1: {2:.3f})\n".format(epoch, final_f1, best_f1_score))
                    # self.log("Results breakdown\n{0}".format(result))
                # if self.updates<200:
                #     # print(self.updates)
                #     self.updates += 1
                #     continue
                self.update(batch)
                if i % 100 == 0:
                    self.log('**********************EPOCH[{0:2}] i[{1:4}] updates[{2:6}] train loss[{3:.5f}] remaining[{4}]'.format(
                        epoch, i, self.updates, self.train_loss.avg,
                        str((datetime.now() - start_time) / (i + 1) * (len(train_batches) - i - 1)).split('.')[0]))

            print("PROGRESS: {0:.2f}%".format(100.0 * (epoch + 1) / num_epochs))
            print('Config file is at ' + self.opt['confFile'])

    def setup_model(self):
        self.train_loss = AverageMeter()
        self.network = ConvQA_CN_Net(self.opt, self.dev_lang)
        if self.use_cuda:
            self.log('Using GPU to setup model...')
            self.network.cuda()
        parameters = [p for p in self.network.parameters() if p.requires_grad]

        ## Ranger优化器
        self.optimizer = Ranger(parameters)
        # self.optimizer = AdamW(parameters, lr=3e-5, weight_decay=0.01)
        self.updates = 0
        self.epoch_start = 0
        # self.loss_func = F.cross_entropy

    def update(self, batch):

        self.network.train()
        self.network.drop_emb = True

        use_his = True
        x, x_offsets, x_bert_mask, rational_mask, x_sep, q, q_mask, ground_truth, context_str, \
            context_word_offsets, ex_pre_answer_strs, is_max_context, token_to_orig_map, answer_types, cls_idx, input_answer_strs, context_ids, turn_ids, his_inf_list, followup_list, yesno_list = batch

        truth = []
        for i in range(len(ground_truth)):
            tmp = torch.LongTensor(ground_truth[i])
            tmp = torch.unsqueeze(tmp, 0)
            truth.append(tmp)
        ground_truth = torch.cat(truth)

        ### forward
        loss = self.network(x, x_bert_mask, rational_mask, x_sep, q, q_mask, his_inf_list, input_answer_strs, ex_pre_answer_strs, ground_truth, context_str, context_ids, turn_ids, answer_types, cls_idx, is_max_context, token_to_orig_map, followup_list, yesno_list, True)

        self.train_loss.update(loss.item(), 1)
        self.optimizer.zero_grad()
        loss.backward()
        # tmp = []
        # for name, p in self.network.named_parameters():
        #     a = p.grad
        #     tmp.append((name, a))

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

        x, x_offsets, x_bert_mask, rational_mask, x_sep, q, q_mask, ground_truth, context_str, \
            context_word_offsets, ex_pre_answer_strs, is_max_context, token_to_orig_map, answer_types, cls_idx, input_answer_strs, context_ids, turn_ids, his_inf_list, followup_list, yesno_list = batch

        pred_json, all_predictions, all_nbest_json= self.network(x, x_bert_mask, rational_mask, x_sep, q, q_mask, his_inf_list, input_answer_strs, ex_pre_answer_strs, ground_truth, context_str, context_ids, turn_ids, answer_types, cls_idx, is_max_context, token_to_orig_map, followup_list, yesno_list, False)

        return pred_json, all_predictions, all_nbest_json


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
