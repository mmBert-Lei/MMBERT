# -*- coding: utf-8 -*-
# ----------------------------------------------------#
#   推理阶段
# ----------------------------------------------------#
# 输入数据
import numpy as np
import torch
from transformers import BertTokenizer

from util import build_gaz_file, decode, NULLKEY, dis2idx
from utils.alphabet import Alphabet
from utils.gazetteer import Gazetteer
from data_help import batchify_with_label, data_initialization
import gc
import os
import sys
import numpy as np
import torch
import time
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import Accuracy, Recall, F1Score, Precision
from data_help import batchify_with_label, data_initialization
from model import Model
from util import decode
from utils.data import Data
import prettytable as pt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
bert_name = 'bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained(bert_name)
gaz = Gazetteer(False)
build_gaz_file(gaz,"D:/nlp/W2NerWithLatticeLSTM10.20/W2NerWithLatticeLSTM/data/ctb.50d.vec")
gaz_alphabet = Alphabet('gaz')
zairu_start = time.time()
model = torch.load('checkpoint/best_model.pt')
zairu_finish = time.time()
zairu_finish = zairu_finish - zairu_start
print('zairu_finish', zairu_finish)
model = model.to(device)
model.eval()
class CFG:
    # n_accumulate = max(1, 32 ) # 延迟更新
    n_accumulate = 1 # 延迟更新
    min_loss = sys.maxsize
    batch_size = 1
    epochs = 2
    min_lr = 3e-4
    T_max = int(50000 / batch_size * epochs) + 50  #
    model_save_path = 'checkpoint/best_model.pt'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data = Data()
    data.HP_gpu = 'gpu'
    data.HP_use_char = True
    data.HP_batch_size = 1
    data.use_bigram = False
    data.gaz_dropout = 0.5
    data.norm_gaz_emb = False
    data.HP_fix_gaz_emb = False
    char_emb = "data/gigaword_chn.all.a2b.uni.ite50.vec"
    bichar_emb = None
    gaz_file = "D:/nlp/W2NerWithLatticeLSTM10.20/W2NerWithLatticeLSTM/data/ctb.50d.vec"
    #train_file, dev_file, test_file = 'data/demo.dev.char', 'data/demo.train.char', 'data/demo.test.char'
    train_file, dev_file, test_file = 'data/cMedQANER/train.bmes','data/cMedQANER/test.bmes', 'data/cMedQANER/ceshi.txt'
    #train_file, dev_file, test_file = 'data/processed1/processed/train_dev.char.bmes', 'data/processed1/processed/dev_dev.char.bmes', 'data/processed1/processed/test.char.bmes'
    data = data_initialization(data, gaz_file, train_file, dev_file, test_file)
def fit_one_test(test_ids,model,test_instence_texts):
    model.eval()
    data_num = len(test_ids)
    total_batch = data_num // CFG.batch_size + 1
    for batch_id in range(total_batch):
        start = batch_id * CFG.batch_size
        end = (batch_id + 1) * CFG.batch_size
        end = data_num if end > data_num else end
        instance = test_ids[start:end]
        if not instance: continue
        gaz_list, batch_word, batch_biword, batch_char, sent_length, grid_mask2d, dist_inputs= batchify_with_label(
            instance, CFG.device)
        with autocast(enabled=True):
            output = model(batch_word, batch_biword, batch_char, gaz_list, sent_length, grid_mask2d, dist_inputs)
            output = output.argmax(axis=-1)
            ent_p, decode_entities = decode(output.cpu().numpy(), sent_length.cpu().numpy())
            print('texts',test_instence_texts[batch_id][0])
            print('ent_p',ent_p)
            print('decode_entities',decode_entities)
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()


if __name__ == '__main__':
    test_ids, test_grid_labels, test_instence_texts = CFG.data.generate_instance_with_gaz(CFG.test_file)
    dev_finish = time.time()
    fit_one_test(test_ids, model, test_instence_texts)
    test_finish = time.time()
    test_cost = test_finish - dev_finish
    print('test_cost', test_cost)