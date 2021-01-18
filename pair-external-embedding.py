# -*- coding: utf-8 -*-
# @Date    : 2021/1/18
# @Author  : mingming.xu
# @Email   : xv44586@gmail.com
# @File    : pair-external-embedding.py
import os
import numpy as np

import numpy as np
from tqdm import tqdm

import gensim

from toolkit4nlp.utils import *
from toolkit4nlp.models import *
from toolkit4nlp.tokenizers import *
from toolkit4nlp.backend import *
from toolkit4nlp.layers import *
from toolkit4nlp.optimizers import *

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
            else:
                label = None
                q_id, r_id, r = span
            D[q_id]['reply'].append([r_id, r, label])
    d = []
    for k, v in D.items():
        q_id = k
        q = v['query']
        reply = v['reply']

        c = []
        l = []
        for r in reply:
            r_id, rc, label = r

            d.append([q_id, q, r_id, rc, label])
    return d


train_data = load_data('train')
test_data = load_data('test')

maxlen = 128
batch_size = 16
epochs = 4
# bert配置
# config_path = '/home/mingming.xu/pretrain/NLP/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = '/home/mingming.xu/pretrain/NLP/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
# dict_path = '/home/mingming.xu/pretrain/NLP/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'

config_path = '/home/mingming.xu/pretrain/NLP/nezha_base_wwm/bert_config.json'
# checkpoint_path = '/home/mingming.xu/pretrain/NLP/nezha_base_wwm/model.ckpt'
dict_path = '/home/mingming.xu/pretrain/NLP/nezha_base_wwm/vocab.txt'
checkpoint_path = './pet_checkpoint/pet_1.model'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    def encode_reply(self, reply_list, label_list):
        tokens, segments = [], []
        mid, r = reply_list[:-1], reply_list[-1]
        mid_l, l = label_list[:-1], label_list[-1]

        # 过滤之前reply 中的 1，防止影响判断
        r_list = []
        for lb, rp in zip(mid_l, mid):
            if lb == 0:
                r_list.append(rp)
            else:
                p = np.random.random()
                if p > 0.:
                    r_list.append(rp)

        # 打乱对话顺序
        np.random.shuffle(r_list)
        for rp in r_list:
            token = tokenizer.encode(rp)[0][1:]
            segment = [0] * len(token)
            tokens += token
            segments += segment

        r_tokens = tokenizer.encode(r)[0][1:]
        r_segments = [1] * len(r_tokens)
        tokens += r_tokens
        segments += r_segments
        return tokens, segments

    def __iter__(self, shuffle=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (q_id, q, r_id, r, label) in self.get_sample(shuffle):
            #             print(q_id, q, r_id, r, label)
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
n = int(len(train_data) * 0.9)
train_generator = data_generator(train_data[:n], batch_size)
valid_generator = data_generator(train_data[n:], batch_size)
test_generator = data_generator(test_data, batch_size)


class ConcatSeq2Vec(Layer):
    def __init__(self, **kwargs):
        super(ConcatSeq2Vec, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ConcatSeq2Vec, self).build(input_shape)

    def call(self, x):
        seq, vec = x
        vec = K.expand_dims(vec, 1)
        vec = K.tile(vec, [1, K.shape(seq)[1], 1])
        return K.concatenate([seq, vec], 2)

    def compute_mask(self, inputs, mask):
        return mask[0]

    def compute_output_shape(self, input_shape):
        return input_shape[0][:-1] + (input_shape[0][-1] + input_shape[1][-1],)


# load w2v
w2v = gensim.models.word2vec.Word2Vec.load('qa_100.w2v')
vocab_list = [(k, w2v.wv[k]) for k, _ in w2v.wv.vocab.items()]
embeddings_matrix = np.zeros((tokenizer._vocab_size, 100))
for (char, vec) in vocab_list:
    embeddings_matrix[tokenizer.encode(char)[0][1:-1]] = vec


class AdaEmbedding(Embedding):
    # 带有可调节学习率的embedding层
    def __init__(self, lr_multiplier=1, **kwargs):
        super(AdaEmbedding, self).__init__(**kwargs)
        self.lr_multiplier = lr_multiplier

    def build(self, input_shape):
        self._embeddings = self.add_weight(
            shape=(self.input_dim, self.output_dim),
            initializer=self.embeddings_initializer,
            name='embeddings',
            regularizer=self.embeddings_regularizer,
            constraint=self.embeddings_constraint,
            dtype=self.dtype)

        self.built = True

        if self.lr_multiplier != 1:
            K.set_value(self._embeddings, K.eval(self._embeddings) / self.lr_multiplier)

    @property
    def embeddings(self):
        if self.lr_multiplier != 1:
            return self._embeddings * self.lr_multiplier
        return self._embeddings

    def call(self, inputs):
        return super(AdaEmbedding, self).call(inputs)


# embedding 层融合
# bert = build_transformer_model(
#     config_path=config_path,
#     checkpoint_path=checkpoint_path,
#     model='nezha',
#     external_embedding_size=100,
#     external_embedding_weights=embeddings_matrix # 融入的embedding
# )

# transformer output层融合
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='nezha',
    #     external_embedding_size=100,
    #     external_embedding_weights=embeddings_matrix
)
output = bert.output

token_input = bert.inputs[0]
ada_embedding = AdaEmbedding(input_dim=tokenizer._vocab_size,
                             name='Embedding-External',
                             output_dim=100,
                             weights=[embeddings_matrix],
                             mask_zero=True,
                             lr_multiplier=2)
external_embedding = ada_embedding(token_input)
x = Concatenate(axis=-1)([output, external_embedding])
output = Lambda(lambda x: x[:, 0])(output)
# output = Dropout(0.1)(output)
output = Dense(1, activation='sigmoid')(output)
model = keras.models.Model(bert.input, output)
model.summary()

optimizer = extend_with_weight_decay(Adam)
optimizer = extend_with_piecewise_linear_lr(optimizer)

opt = optimizer(learning_rate=1e-5,
                weight_decay_rate=0.05, exclude_from_weight_decay=['Norm', 'bias'],
                lr_schedule={int(len(train_generator) * epochs * 0.1): 1, len(train_generator) * epochs: 0})

model.compile(
    #     loss=binary_focal_loss(0.25, 12),
    loss=K.binary_crossentropy,
    optimizer=opt,
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
        if ada_embedding.lr_multiplier != 1:
            ada_embedding.lr_multiplier = ada_embedding.lr_multiplier * 0.9

        val_f1 = evaluate(valid_generator)
        if val_f1 > self.best_val_f1:
            self.best_val_f1 = val_f1
            model.save_weights('best_parimatch_model.weights')
        #         test_acc = evaluate(test_generator)
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
        epochs=epochs,
        callbacks=[evaluator],
        #     class_weight={0:1, 1:4}
    )
    # load best model and predict result
    model.load_weights('best_parimatch_model.weights')
    predict_to_file()
