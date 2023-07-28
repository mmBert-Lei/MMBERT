# -*- coding: utf-8 -*-
# ----------------------------------------------------#
#   模型训练阶段
# ----------------------------------------------------#
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
from gensim.models import Word2Vec
from sklearn.metrics import precision_recall_fscore_support, f1_score

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
class CFG:
    # n_accumulate = max(1, 32 ) # 延迟更新
    n_accumulate = 1 # 延迟更新
    min_loss = sys.maxsize
    batch_size = 1
    epochs = 20
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
    gaz_file = "data/ctb.50d.vec"
    #train_file, dev_file, test_file = 'data/demo.dev.char', 'data/demo.train.char', 'data/demo.test.char'
    train_file, dev_file, test_file = 'data/cMedQANER/train.bmes','data/cMedQANER/test.bmes', 'data/cMedQANER/test.bmes'
    #train_file, dev_file, test_file = 'data/processed1/processed/train_dev.char.bmes', 'data/processed1/processed/dev_dev.char.bmes', 'data/processed1/processed/test.char.bmes'
    data = data_initialization(data, gaz_file, train_file, dev_file, test_file)


model = Model(CFG.data).to(CFG.device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = AdamW(params)
scheduler = CosineAnnealingLR(optimizer, T_max=CFG.T_max,
                              eta_min=CFG.min_lr)
def fit_one_train(train_ids,grid_labels,model,optimizer,scheduler,epoch):
    model.train()
    epoch_start = time.time()
    temp_start = epoch_start
    total_loss = 0
    train_num = len(train_ids)
    total_batch = train_num // CFG.batch_size + 1
    scaler = GradScaler()
    criterion = CrossEntropyLoss().to(CFG.device)
    for batch_id in range(total_batch):
        start = batch_id * CFG.batch_size
        end = (batch_id + 1) * CFG.batch_size
        end = train_num if end > train_num else end
        instance = train_ids[start:end]
        grid_labe = grid_labels[start:end]
        grid_label = torch.LongTensor(np.array(grid_labe)).to(CFG.device)
        if not instance: continue
        gaz_list, batch_word, batch_biword, batch_char, sent_length, grid_mask2d, dist_inputs= batchify_with_label(
            instance, CFG.device)
        with autocast(enabled=True):
            output = model(batch_word, batch_biword, batch_char, gaz_list, sent_length, grid_mask2d, dist_inputs)
            grid_mask2d = grid_mask2d.clone()
            # print(output.shape)
            # print(grid_label.shape)
            # print(grid_mask2d.shape)
            loss = criterion(output[grid_mask2d], grid_label[grid_mask2d])
            loss = loss / CFG.n_accumulate
        scaler.scale(loss).backward()
        if (batch_id + 1) % CFG.n_accumulate == 0:
            scaler.step(optimizer)
            scaler.update()
            # zero the parameter gradients
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()
        total_loss += loss.item()
        # 梯度优化反向传播更新权值
    scheduler.step()
    #print(f'Train-Loss({epoch}):',total_loss)
    epoch_finish = time.time()
    epoch_cost = epoch_finish - epoch_start
    print('epoch_cost=',epoch_cost)
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()

eps = 1e-6
def get_f1_score_label(preds, labels, label="organization"):
    """
    pr指标计算
    """
    TP = 0
    FP = 0
    FN = 0
    for pre, gold in zip(preds, labels):
        if pre > 1:
            if pre == gold:
                TP += 1
            else:
                FP += 1
        if gold > 1:
            if pre <= 1:
                FN += 1
    p = TP / (TP + FP + eps)
    r = TP / (TP + FN + eps)
    f = 2 * p * r / (p + r + eps)
    return p, r, f

@torch.no_grad()
def fit_one_val(val_ids, grid_labels,model,epoch):
    model.eval()
    total_loss = 0
    pred_result = []
    label_result = []
    data_num = len(val_ids)
    total_batch = data_num // CFG.batch_size + 1
    criterion = CrossEntropyLoss().to(CFG.device)
    #acc = Accuracy(num_classes=CFG.data.type_len,average = 'weighted')
    #recall = Recall(num_classes=CFG.data.type_len,average = 'weighted')
    #f1_score = F1Score(num_classes=CFG.data.type_len,average = 'weighted')
    #precision = Precision(num_classes=CFG.data.type_len,average = 'weighted')
    for batch_id in range(total_batch):
        # print(f'{batch_id}/{total_batch}')
        start = batch_id * CFG.batch_size
        end = (batch_id + 1) * CFG.batch_size
        end = data_num if end > data_num else end
        instance = val_ids[start:end]
        grid_labe = grid_labels[start:end]
        grid_label = torch.LongTensor(np.array(grid_labe)).to(CFG.device)
        if not instance: continue
        gaz_list, batch_word, batch_biword, batch_char, sent_length, grid_mask2d, dist_inputs = batchify_with_label(
            instance, CFG.device)
        with autocast(enabled=True):
            output =  model(batch_word, batch_biword, batch_char, gaz_list, sent_length, grid_mask2d, dist_inputs)
            output = output[grid_mask2d]
            grid_label = grid_label[grid_mask2d]
            loss = criterion(output, grid_label)
        outputs_one = output.argmax(axis=-1)
        #grid_labels = grid_label.contiguous().view(-1)
        #outputs = outputs_one.contiguous().view(-1)
        #acc.update(grid_label.cpu(),outputs_one.cpu())
        #recall.update(grid_label.cpu(),outputs_one.cpu())
        #f1_score.update(grid_label.cpu(),outputs_one.cpu())
        #precision.update(grid_label.cpu(),outputs_one.cpu())
        grid_label = grid_label.contiguous().view(-1)
        outputs_one = outputs_one.contiguous().view(-1)
        label_result.append(grid_label.cpu())
        pred_result.append(outputs_one.cpu())
        batch_id += CFG.batch_size
        total_loss += loss.item()
    label_result = torch.cat(label_result)
    pred_result = torch.cat(pred_result)
    p, r, f1 = get_f1_score_label(pred_result.numpy(), label_result.numpy())
    #np.save("label.npy", label_result.numpy())
    #np.save("predict.npy", pred_result.numpy())

    title = "dev"

    table = pt.PrettyTable(["{} {}".format(title, epoch), 'F1', "Precision", "Recall"])
    table.add_row(["Label"] + ["{:3.4f}".format(x) for x in [f1, p, r]])

    print("\n{}".format(table))
    #print(f'Val-Loss({epoch}):', total_loss)
    #print(f'Val-Acc({epoch}):', acc.compute())
    #print(f'Val-Recall({epoch}):', recall.compute())
    #print(f'Val-F1_score({epoch}):', f1_score.compute())
    #print(f'Val-Precision({epoch}):', precision.compute())
    if total_loss < CFG.min_loss:
        # 保存模型
        torch.save(model, CFG.model_save_path)
        CFG.min_loss = total_loss
        print(CFG.min_loss)
    #acc.reset()
    #r.reset()
    #f1.reset()
    #p.reset()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
@torch.no_grad()
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
    epoch_finish = time.time()
    for epoch in range(CFG.epochs):
        train_ids, train_grid_labels, _ = CFG.data.generate_instance_with_gaz(CFG.train_file)
        fit_one_train(train_ids, train_grid_labels, model, optimizer, scheduler, epoch)
        del train_ids, train_grid_labels
        gc.collect()
        val_ids, val_grid_labels,_ = CFG.data.generate_instance_with_gaz(CFG.test_file)
        fit_one_val(val_ids, val_grid_labels,model,epoch)
    dev_finish = time.time()
    dev_cost = dev_finish - epoch_finish
    print('dev_cost=', dev_cost)
        # del val_ids, val_grid_labels
        # gc.collect()
    # 测试
    test_ids, test_grid_labels, test_instence_texts = CFG.data.generate_instance_with_gaz(CFG.test_file)
    dev_finish = time.time()
    fit_one_test(test_ids, model, test_instence_texts)
    test_finish = time.time()
    test_cost = test_finish - dev_finish
    print('test_cost', test_cost)

    del test_ids, test_grid_labels, test_instence_texts
    gc.collect()
