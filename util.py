from collections import defaultdict, deque

import numpy as np

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
NULLKEY = "-null-"
def convert_index_to_text(index, type):
    text = "-".join([str(i) for i in index])
    text = text + "-#-{}".format(type)
    return text


def convert_text_to_index(text):
    index, type = text.split("-#-")
    index = [int(x) for x in index.split("-")]
    return index, int(type)


def decode(outputs, length):
    class Node:
        def __init__(self):
            self.THW = []  # [(tail, type)]
            self.NNW = defaultdict(set)  # {(head,tail): {next_index}}

    ent_r, ent_p, ent_c = 0, 0, 0
    decode_entities = []
    q = deque()  # 集合
    # zip 后接多个迭代器 前面为迭代出来的东西
    # '0-1-#-2' 01 是2类  '3-4-5-#-2' 345是2类
    for instance, l in zip(outputs, length):
        predicts = []
        # 七个词七个节点
        nodes = [Node() for _ in range(l)]
        # 把(7,6,5,4,3,2,1)
        for cur in reversed(range(l)):
            heads = []
            for pre in range(cur + 1):
                # THW(下一个相邻单词)
                if instance[cur, pre] > 1:
                    nodes[pre].THW.append((cur, instance[cur, pre]))
                    heads.append(pre)
                # NNW(尾部单词)
                if pre < cur and instance[pre, cur] == 1:
                    # cur node
                    for head in heads:
                        nodes[pre].NNW[(head, cur)].add(cur)
                    # post nodes
                    for head, tail in nodes[cur].NNW.keys():
                        if tail >= cur and head <= pre:
                            nodes[pre].NNW[(head, tail)].add(cur)
            # entity
            for tail, type_id in nodes[cur].THW:
                if cur == tail:
                    predicts.append(([cur], type_id))
                    continue
                q.clear()
                q.append([cur])
                while len(q) > 0:
                    chains = q.pop()
                    for idx in nodes[chains[-1]].NNW[(cur, tail)]:
                        if idx == tail:
                            predicts.append((chains + [idx], type_id))
                        else:
                            q.append(chains + [idx])

        predicts = set([convert_index_to_text(x[0], x[1]) for x in predicts])
        decode_entities.append([convert_text_to_index(x) for x in predicts])

        ent_p += len(predicts)
    return ent_p,decode_entities
def build_gaz_file(gaz,gaz_file):
    if gaz_file:
        fins = open(gaz_file, 'r', encoding='utf-8').readlines()
        for fin in fins:
            fin = fin.strip().split()[0]
            if fin:
                gaz.insert(fin, "one_source")
        print("Load gaz file: ", gaz_file, " total size:", gaz.size())
    else:
        print("Gaz file is None, load nothing")