#!usr/bin/env python
# encoding: utf-8

#from const.paths import msra_ner_cn_path
import os
import sys

def BIO2BMES(input_file, output_file):
    print("Convert BIO -> BMES for file:", input_file)
    with open(input_file,'r',encoding='UTF-8') as in_file:
        fins = in_file.readlines()
    fout = open(output_file,'w',encoding='UTF-8')
    words = []
    labels = []
    for line in fins:
        if len(line) < 3:
            sent_len = len(words)
            for idx in range(sent_len):
                if "-" not in labels[idx]:
                    fout.write(words[idx]+" "+labels[idx]+"\n")
                else:
                    label_type = labels[idx].split('-')[-1]
                    if "B-" in labels[idx]:
                        if (idx == sent_len - 1) or ("I-" not in labels[idx+1]):
                            fout.write(words[idx]+" S-"+label_type+"\n")
                        else:
                            fout.write(words[idx]+" B-"+label_type+"\n")
                    elif "I-" in labels[idx]:
                        if (idx == sent_len - 1) or ("I-" not in labels[idx+1]):
                            fout.write(words[idx]+" E-"+label_type+"\n")
                        else:
                            fout.write(words[idx]+" M-"+label_type+"\n")
            fout.write('\n')
            words = []
            labels = []
        else:
            if line == '0\t\n':
                words.append('0')
                labels.append('O')
            else:
                pair = line.strip('\n').split()
                words.append(pair[0])
                labels.append(pair[-1].upper())
    fout.close()
    print("BMES file generated:", output_file)

def msra_bio2bmes(msrapath):
    train_dev_path = os.path.join(msrapath, 'train.txt')
    train_dev_path_out = os.path.join(msrapath, 'train_dev.char.bmes')
    dev_dev_path = os.path.join(msrapath, 'dev.txt')
    dev_dev_path_out = os.path.join(msrapath, 'dev_dev.char.bmes')
    test_path = os.path.join(msrapath, 'test.txt')
    test_path_out = os.path.join(msrapath, 'test.char.bmes')

    BIO2BMES(train_dev_path, train_dev_path_out)
    BIO2BMES(dev_dev_path, dev_dev_path_out)
    BIO2BMES(test_path, test_path_out)

if __name__ == '__main__':
    msra_bio2bmes('data/processedccks2019/')
    print('- Done!')

