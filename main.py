# -- coding:UTF-8 --

import sys
import os
import logging
import torch
import argparse
from Models.ConvQA_CN_NetTrainer import ConvQA_CN_NetTrainer
from Utils.Arguments import Arguments


def main():
    #加载日志
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger = logging.getLogger(__name__)

    # 'opt' means options
    opt = None

    parser = argparse.ArgumentParser(description='multi-qa')
    parser.add_argument('mode', help='mode: train')
    parser.add_argument('conf_file', help='path to conf file.')
    parser.add_argument('dataset', help='dataset for train')

    #得到参数实例
    args = parser.parse_args()
    mode = args.mode # train mode

    #得到conf文件的参数
    conf_file = args.conf_file
    dataset = args.dataset
    conf_args = Arguments(conf_file)
    opt = conf_args.readArguments()

    opt['cuda'] = torch.cuda.is_available()
    opt['confFile'] = conf_file
    opt['datadir'] = os.path.dirname(conf_file) # .
    for key, val in args.__dict__.items():
        if val is not None and key not in ['command', 'conf_file']:
            opt[key] = val

    #使用GPU
    # opt['cuda'] = False
    device = torch.device("cuda" if opt['cuda'] else 'cpu')
    logger.info("device %s is using for training!", device)
    model = ConvQA_CN_NetTrainer(opt)
    print("Select mode----", mode)
    print("Using dataset ----", dataset)
    model.train()
if __name__ == "__main__":
    main()