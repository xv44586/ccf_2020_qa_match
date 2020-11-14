# -*- coding: utf-8 -*-
# @Date    : 2020/11/4
# @Author  : mingming.xu
# @Email   : xv44586@gmail.com
# @File    : ccf_2020_qa_match_pet.py
"""
Pattern-Exploiting Training(PET): 增加pattern，将任务转换为MLM任务。
线上f1: 0.761

tips:
  切换模型时，修改对应config_path/checkpoint_path/dict_path路径以及build_transformer_model 内的参数
"""

import os
import numpy as np
import json
from tqdm import tqdm
import numpy as np

from toolkit4nlp.backend import keras, K
from toolkit4nlp.tokenizers import Tokenizer, load_vocab
from toolkit4nlp.models import build_transformer_model, Model
from toolkit4nlp.optimizers import *
from toolkit4nlp.utils import pad_sequences, DataGenerator
from toolkit4nlp.layers import *

path = '/home/mingming.xu/datasets/NLP/ccf_qa_match/'

p = os.path.join(path, 'train', 'train.query.tsv')


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
                label = int(label)
            else:
                label = None
                q_id, r_id, r = span
            D[q_id]['reply'].append([r_id, r, label])
    d = []
    for k, v in D.items():
        q_id = k
        q = v['query']
        reply = v['reply']

        for i, r in enumerate(reply):
            r_id, rc, label = r
            d.append([q_id, q, r_id, rc, label])
    return d


train_data = load_data('train')
test_data = load_data('test')

num_classes = 32
maxlen = 128
batch_size = 8

# BERT base

config_path = '/home/mingming.xu/pretrain/NLP/nezha_base_wwm/bert_config.json'
checkpoint_path = '/home/mingming.xu/pretrain/NLP/nezha_base_wwm/model.ckpt'
dict_path = '/home/mingming.xu/pretrain/NLP/nezha_base_wwm/vocab.txt'

# tokenizer
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# pattern
pattern = '直接回答问题:'
mask_idx = [1]

id2label = {
    0: '间',
    1: '直'
}

label2id = {v: k for k, v in id2label.items()}
labels = list(id2label.values())


def random_masking(token_ids):
    """对输入进行随机mask
    """
    rands = np.random.random(len(token_ids))
    source, target = [], []
    for r, t in zip(rands, token_ids):
        if r < 0.15 * 0.8:
            source.append(tokenizer._token_mask_id)
            target.append(t)
        elif r < 0.15 * 0.9:
            source.append(t)
            target.append(t)
        elif r < 0.15:
            source.append(np.random.choice(tokenizer._vocab_size - 1) + 1)
            target.append(t)
        else:
            source.append(t)
            target.append(0)
    return source, target


class data_generator(DataGenerator):
    def __init__(self, prefix=False, *args, **kwargs):
        super(data_generator, self).__init__(*args, **kwargs)
        self.prefix = prefix

    def __iter__(self, shuffle=False):
        batch_token_ids, batch_segment_ids, batch_target_ids = [], [], []

        for is_end, (q_id, q, r_id, r, label) in self.get_sample(shuffle):
            label = int(label) if label is not None else None

            if label is not None or self.prefix:
                q = pattern + q

            token_ids, segment_ids = tokenizer.encode(q, r, maxlen=maxlen)

            if shuffle:
                source_tokens, target_tokens = random_masking(token_ids)
            else:
                source_tokens, target_tokens = token_ids[:], token_ids[:]

            # mask label
            if label is not None:
                label_ids = tokenizer.encode(id2label[label])[0][1:-1]
                for m, lb in zip(mask_idx, label_ids):
                    source_tokens[m] = tokenizer._token_mask_id
                    target_tokens[m] = lb
            elif self.prefix:
                for i in mask_idx:
                    source_tokens[i] = tokenizer._token_mask_id

            batch_token_ids.append(source_tokens)
            batch_segment_ids.append(segment_ids)
            batch_target_ids.append(target_tokens)

            if is_end or len(batch_token_ids) == self.batch_size:
                batch_token_ids = pad_sequences(batch_token_ids)
                batch_segment_ids = pad_sequences(batch_segment_ids)
                batch_target_ids = pad_sequences(batch_target_ids)

                yield [batch_token_ids, batch_segment_ids, batch_target_ids], None

                batch_token_ids, batch_segment_ids, batch_target_ids = [], [], []


# shuffle
np.random.shuffle(train_data)
n = int(len(train_data) * 0.8)
train_generator = data_generator(data=train_data[: n] + test_data, batch_size=batch_size)
valid_generator = data_generator(data=train_data[n:], batch_size=batch_size)
test_generator = data_generator(data=test_data, batch_size=batch_size, prefix=True)


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """

    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
        accuracy = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        accuracy = K.sum(accuracy * y_mask) / K.sum(y_mask)
        self.add_metric(accuracy, name='accuracy')
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


model = build_transformer_model(config_path=config_path,
                                checkpoint_path=checkpoint_path,
                                with_mlm=True,
                                # model='bert',  # 加载bert/Roberta/ernie
                                model='nezha'
                                )

target_in = Input(shape=(None,))
output = CrossEntropy(1)([target_in, model.output])

train_model = Model(model.inputs + [target_in], output)

AdamW = extend_with_weight_decay(Adam)
AdamWG = extend_with_gradient_accumulation(AdamW)

opt = AdamWG(learning_rate=1e-5, exclude_from_weight_decay=['Norm', 'bias'], grad_accum_steps=4)
train_model.compile(opt)
train_model.summary()

label_ids = np.array([tokenizer.encode(l)[0][1:-1] for l in labels])


def predict(x):
    if len(x) == 3:
        x = x[:2]
    y_pred = model.predict(x)[:, mask_idx]
    y_pred = y_pred[:, 0, label_ids[:, 0]]
    y_pred = y_pred.argmax(axis=1)
    return y_pred


def evaluate(data):
    P, R, TP = 0., 0., 0.
    for d, _ in tqdm(data):
        x_true, y_true = d[:2], d[2]

        y_pred = predict(x_true)
        y_true = np.array([labels.index(tokenizer.decode(y)) for y in y_true[:, mask_idx]])
        #         print(y_true, y_pred)
        R += y_pred.sum()
        P += y_true.sum()
        TP += ((y_pred + y_true) > 1).sum()

    print(P, R, TP)
    pre = TP / R
    rec = TP / P

    return 2 * (pre * rec) / (pre + rec)


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        acc = evaluate(valid_generator)
        if acc > self.best_acc:
            self.best_acc = acc
            self.model.save_weights('best_pet_model.weights')
        print('acc :{}, best acc:{}'.format(acc, self.best_acc))


def write_to_file(path):
    preds = []
    for x, _ in tqdm(test_generator):
        pred = predict(x)
        preds.extend(pred)

    ret = []
    for data, p in zip(test_data, preds):
        ret.append([data[0], data[2], str(p)])

    with open(path, 'w') as f:
        for r in ret:
            f.write('\t'.join(r) + '\n')


if __name__ == '__main__':
    evaluator = Evaluator()
    train_model.fit_generator(train_generator.generator(),
                              steps_per_epoch=len(train_generator),
                              epochs=10,
                              callbacks=[evaluator])

    train_model.load_weights('best_pet_model.weights')
    write_to_file('submission.tsv')
