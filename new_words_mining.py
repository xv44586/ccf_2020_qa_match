# -*- coding: utf-8 -*-
# @Date    : 2020/12/8
# @Author  : mingming.xu
# @Email   : xv44586@gmail.com
# @File    : pmi.py
"""
根据PMI 挖掘新词与短语
ref: [最小熵原理（二）：“当机立断”之词库构建](https://kexue.fm/archives/5476)
"""
import os
from nlp_zero import *
import jieba

jieba.initialize()
path = '/home/mingming.xu/datasets/NLP/ccf_qa_match/'


def load_data(train_test='train'):
    D = {}
    with open(os.path.join(path, train_test, train_test + '.query.tsv')) as f:
        for l in f:
            span = l.strip().split('\t')
            D[span[0]] = {'query': span[1], 'reply': []}

    with open(os.path.join(path, train_test, train_test + '.reply.tsv')) as f:
        for l in f:
            span = l.strip().split('\t')
            if len(span) == 4:
                q_id, r_id, r, label = span
            else:
                label = None
                q_id, r_id, r = span
            D[q_id]['reply'].append([r_id, r, label])
    d = []
    for k, v in D.items():
        q = v['query']
        reply = v['reply']

        cor = [q] + [r[1] for r in reply]
        d.append(''.join(cor))

    return d


train_data = load_data('train')
test_data = load_data('test')


class G(object):
    def __iter__(self):
        for i in train_data + test_data:
            yield i


f = Word_Finder(min_proba=1e-5)
f.train(G())
f.find(G())

# 长度为2~5 且不在jieba 词典内的词
new_words = [w for w, _ in f.words.items() if len(w) > 2 and len(w) < 5 and len(jieba.lcut(w, HMM=False)) > 1]

with open('new_dict.txt', 'w') as f:
    f.write('\n'.join(new_words))
