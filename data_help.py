# -*- coding: utf-8 -*-
# ----------------------------------------------------#
#   数据处理模块
# ----------------------------------------------------#
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import torch.autograd as autograd

dis2idx = np.zeros((1000), dtype='int64')
dis2idx[1] = 1
dis2idx[2:] = 2
dis2idx[4:] = 3
dis2idx[8:] = 4
dis2idx[16:] = 5
dis2idx[32:] = 6
dis2idx[64:] = 7
dis2idx[128:] = 8
dis2idx[256:] = 9
class RelationDataset(Dataset):
    def __init__(self, texts, ids, grid_labels, grid_mask2d, dist_inputs, pieces2word, sent_length):
        self.texts = texts
        self.grid_labels = grid_labels
        self.grid_mask2d = grid_mask2d
        self.pieces2word = pieces2word
        self.dist_inputs = dist_inputs
        self.sent_length = sent_length
        self.ids = ids

    def __getitem__(self, item):
        return self.grid_labels[item],self.grid_mask2d[item],self.pieces2word[item],self.dist_inputs[item],self.sent_length[item]
    def __len__(self):
        return len(self.grid_labels)
def batchify_with_label(input_batch_list,device='cpu',volatile_flag=False):
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]  # 1-gram(chars) sequence
    biwords = [sent[1] for sent in input_batch_list]  # 2-gram
    chars = [sent[2] for sent in input_batch_list]  # 1-gram list
    gazs = [sent[3] for sent in input_batch_list]  # possible words/terms
    labels = [sent[4] for sent in input_batch_list]  # label sequence
    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max()  # max sentence's length(characters) in the batch (batch_size =1)
    word_seq_tensor = torch.zeros((batch_size, max_seq_len)).long()
    biword_seq_tensor = torch.zeros((batch_size, max_seq_len)).long()
    dist_inputs = []
    grid_mask2d = []
    grid_mask2d = []
    sent_length = []
    for idx, (seq, biseq, label, seqlen) in enumerate(zip(words, biwords, labels, word_seq_lengths)):
        sent_length.append(seqlen)
        _dist_inputs = np.zeros((seqlen, seqlen), dtype=np.int)
        _grid_mask2d = np.ones((seqlen, seqlen), dtype=np.bool)
        grid_mask2d.append(_grid_mask2d)
        for k in range(seqlen):
            _dist_inputs[k, :] += k
            _dist_inputs[:, k] -= k

        for i in range(seqlen):
            for j in range(seqlen):
                if _dist_inputs[i, j] < 0:
                    _dist_inputs[i, j] = dis2idx[-_dist_inputs[i, j]] + 9
                else:
                    _dist_inputs[i, j] = dis2idx[_dist_inputs[i, j]]
        _dist_inputs[_dist_inputs == 0] = 19
        dist_inputs.append(_dist_inputs)
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        biword_seq_tensor[idx, :seqlen] = torch.LongTensor(biseq)
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0,
                                                            descending=True)  # char sequence length in the batch is descending
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    biword_seq_tensor = biword_seq_tensor[word_perm_idx]
    ## not reorder label
    pad_chars = [chars[idx] + np.zeros(max_seq_len - len(chars[idx])) if max_seq_len > len(chars[idx]) else chars[idx]
                 for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]

    max_word_len = max(map(max, length_list))  # 1
    char_seq_tensor = autograd.Variable(torch.zeros((batch_size, max_seq_len, max_word_len))).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):

        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            # print len(word), wordlen
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)
    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size * max_seq_len, -1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size * max_seq_len, )
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)

    ## keep the gaz_list in orignial order

    gaz_list = [gazs[i] for i in word_perm_idx]
    gaz_list.append(volatile_flag)
    sent_length = torch.tensor(sent_length)
    word_seq_tensor = word_seq_tensor.to(device)  # 1-gram sequence [1, seq_length]
    biword_seq_tensor = biword_seq_tensor.to(device)  # 2-gram sequence
    char_seq_tensor = char_seq_tensor.to(device)
    sent_length = sent_length.to(device)  # all the elements is 1.
    grid_mask2d = torch.tensor(np.array(grid_mask2d)).to(device)
    dist_inputs = torch.LongTensor(np.array(dist_inputs)).to(device)
    return gaz_list, word_seq_tensor, biword_seq_tensor, char_seq_tensor,sent_length,grid_mask2d,dist_inputs
def data_initialization(data, gaz_file, train_file, dev_file, test_file):
    data.build_alphabet(train_file)
    data.build_alphabet(dev_file)
    data.build_alphabet(test_file)
    data.build_gaz_file(gaz_file)
    data.build_gaz_alphabet(train_file)
    data.build_gaz_alphabet(dev_file)
    data.build_gaz_alphabet(test_file)
    data.fix_alphabet()
    print('类别对应关系',data.type2id)
    data.type_len = data.type_index+1
    return data