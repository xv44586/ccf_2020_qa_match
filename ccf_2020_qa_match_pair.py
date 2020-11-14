# -*- coding: utf-8 -*-
# @Date    : 2020/11/3
# @Author  : mingming.xu
# @Email   : xv44586@gmail.com
# @File    : ccf_2020_qa_match_pair.py
"""
拆成query-pair 对，然后分类
线上f1:0.752

tips:
  切换模型时，修改对应config_path/checkpoint_path/dict_path路径以及build_transformer_model 内的参数
"""
import os
from tqdm import tqdm
import numpy as np

from toolkit4nlp.utils import *
from toolkit4nlp.models import *
from toolkit4nlp.layers import *
from toolkit4nlp.optimizers import *
from toolkit4nlp.tokenizers import Tokenizer
from toolkit4nlp.backend import *

batch_size = 16
maxlen = 280
epochs = 10
lr = 1e-5

# bert配置
config_path = '/home/mingming.xu/pretrain/NLP/nezha_base_wwm/bert_config.json'
checkpoint_path = '/home/mingming.xu/pretrain/NLP/nezha_base_wwm/model.ckpt'
dict_path = '/home/mingming.xu/pretrain/NLP/nezha_base_wwm//vocab.txt'
# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

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
        q_id = k
        q = v['query']
        reply = v['reply']

        for r in reply:
            r_id, rc, label = r

            d.append([q_id, q, r_id, rc, label])
    return d


train_data = load_data('train')
test_data = load_data('test')


class data_generator(DataGenerator):
    def __iter__(self, shuffle=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (q_id, q, r_id, r, label) in self.get_sample(shuffle):
            label = int(label) if label is not None else None

            token_ids, segment_ids = tokenizer.encode(q, r, maxlen=256)

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])

            if is_end or len(batch_token_ids) == self.batch_size:
                batch_token_ids = pad_sequences(batch_token_ids)
                batch_segment_ids = pad_sequences(batch_segment_ids)
                batch_labels = pad_sequences(batch_labels)

                yield [batch_token_ids, batch_segment_ids], batch_labels

                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# shuffle
np.random.shuffle(train_data)
n = int(len(train_data) * 0.8)
train_generator = data_generator(train_data[:n], batch_size)
valid_generator = data_generator(train_data[n:], batch_size)
test_generator = data_generator(test_data, batch_size)

# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    # model='bert',  # 加载bert/Roberta/ernie
    # model='electra', # 加载electra
    model='nezha',  # 加载NEZHA
)
output = bert.output

output = Dropout(0.5)(output)

att = AttentionPooling1D(name='attention_pooling_1')(output)

output = ConcatSeq2Vec()([output, att])
output = DGCNN(dilation_rate=1, dropout_rate=0.1)(output)
output = DGCNN(dilation_rate=2, dropout_rate=0.1)(output)
output = DGCNN(dilation_rate=5, dropout_rate=0.1)(output)
output = Lambda(lambda x: x[:, 0])(output)
output = Dense(1, activation='sigmoid')(output)

model = keras.models.Model(bert.input, output)
model.summary()

model.compile(
    loss=K.binary_crossentropy,
    optimizer=Adam(2e-5),
    metrics=['accuracy'],
)

model.compile(
    loss=K.binary_crossentropy,
    optimizer=Adam(2e-5),  # 用足够小的学习率
    metrics=['accuracy'],
)


def evaluate(data):
    P, R, TP = 0., 0., 0.
    for x_true, y_true in tqdm(data):
        y_pred = model.predict(x_true)[:, 0]
        y_pred = np.round(y_pred)
        y_true = y_true[:, 0]

        R += y_pred.sum()
        P += y_true.sum()
        TP += ((y_pred + y_true) > 1).sum()

    print(P, R, TP)
    pre = TP / R
    rec = TP / P

    return 2 * (pre * rec) / (pre + rec)


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """

    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_f1 = evaluate(valid_generator)
        if val_f1 > self.best_val_f1:
            self.best_val_f1 = val_f1
            model.save_weights('best_parimatch_model.weights')
        print(
            u'val_f1: %.5f, best_val_f1: %.5f\n' %
            (val_f1, self.best_val_f1)
        )


def predict_to_file(path='pair_submission.tsv', data=test_generator):
    preds = []
    for x, _ in tqdm(test_generator):
        pred = model.predict(x)[:, 0]
        pred = np.round(pred)
        pred = pred.astype(int)
        preds.extend(pred)

    ret = []
    for d, p in zip(test_data, preds):
        q_id, _, r_id, _, _ = d
        ret.append([str(q_id), str(r_id), str(p)])

    with open(path, 'w', encoding='utf8') as f:
        for l in ret:
            f.write('\t'.join(l) + '\n')


if __name__ == '__main__':
    evaluator = Evaluator()
    model.fit_generator(
        train_generator.generator(),
        steps_per_epoch=len(train_generator),
        epochs=5,
        callbacks=[evaluator],
    )

    # predict test and write to file
    model.load_weights('best_parimatch_model.weights')
    predict_to_file()
