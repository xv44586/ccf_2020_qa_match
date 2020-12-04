# -*- coding: utf-8 -*-
# @Date    : 2020/12/1
# @Author  : mingming.xu
# @Email   : xv44586@gmail.com
# @File    : post-training-wwm-sop.py
"""
训练样本格式为：[query, reply1, reply2,..], 此外替换NSP 为SOP，且SOP 时只替换reply list 顺序
"""
import os

os.environ['TF_KERAS'] = '1'  # 必须使用tf.keras

import numpy as np
from tqdm import tqdm
import jieba
import itertools

from toolkit4nlp.utils import DataGenerator, pad_sequences
from toolkit4nlp.models import *
from toolkit4nlp.tokenizers import *
from toolkit4nlp.backend import *
from toolkit4nlp.layers import *
from toolkit4nlp.optimizers import *

# config
path = '/home/mingming.xu/datasets/NLP/ccf_qa_match/'

p = os.path.join(path, 'train', 'train.query.tsv')
config_path = '/home/mingming.xu/pretrain/NLP/nezha_base_wwm/bert_config.json'
checkpoint_path = '/home/mingming.xu/pretrain/NLP/nezha_base_wwm/model.ckpt'
dict_path = '/home/mingming.xu/pretrain/NLP/nezha_base_wwm/vocab.txt'
model_saved_path = './nezha_post_training/wwm-model-add-dict-no-mask-end-sop.ckpt'

new_dict_path = './new_dict.txt'
maxlen = 256
batch_size = 16
epochs = 100
learning_rate = 5e-5
# 建立分词器
tokenizer = Tokenizer(dict_path)


# query /reply 拼接作为训练句子
def load_data(train_test='train'):
    D = {}
    with open(os.path.join(path, train_test, train_test + '.query.tsv')) as f:
        for l in f:
            span = l.strip().split('\t')
            q = span[1]
            D[span[0]] = {'query': q, 'reply': []}

    with open(os.path.join(path, train_test, train_test + '.reply.tsv')) as f:
        for l in f:
            span = l.strip().split('\t')
            if len(span) == 4:
                q_id, r_id, r, label = span
            else:
                q_id, r_id, r = span

            #             if len(r) < 4 or (len(r) == 1 and tokenizer._is_punctuation(r)):
            #                 continue

            #             补上句号
            #             if not tokenizer._is_punctuation(list(r)[-1]):
            #                 r += '。'

            D[q_id]['reply'].append(r)

    d = []
    for k, v in D.items():
        item = []
        l = 0
        q = v['query']
        replys = v['reply']
        l += len(q)
        item.append(q)
        for r in replys:
            lr = len(r)
            #             if l + lr >maxlen:
            #                 d.append(item)
            #                 item = []
            #                 l = 0

            l += lr
            item.append(r)

        d.append(item)

    return d


train_data = load_data('train')
test_data = load_data('test')
data = train_data + test_data

# wwm
jieba.initialize()

new_words = []
with open(new_dict_path) as f:
    for l in f:
        w = l.strip()
        new_words.append(w)
        jieba.add_word(w)

words_data = [[jieba.lcut(line) for line in sen] for sen in data]


def shuffle_reply(item):
    """
    只打乱reply list的顺序
    """
    q, rs = item[0], item[1:]
    permuter_rs = list(itertools.permutations(rs))[1:]
    if len(permuter_rs) < 1:
        print(item)
    idx = np.random.choice(len(permuter_rs))
    r = permuter_rs[idx]
    return [q] + list(r)


def can_mask(token_ids):
    if token_ids in (tokenizer._token_start_id, tokenizer._token_mask_id, tokenizer._token_end_id):
        return False

    return True


def random_masking(lines):
    """对输入进行随机mask
    """

    #     rands = np.random.random(len(token_ids))
    sources, targets = [tokenizer._token_start_id], [0]
    segments = [0]

    for i, sent in enumerate(lines):
        source, target = [], []
        segment = []
        rands = np.random.random(len(sent))
        for r, word in zip(rands, sent):
            word_token = tokenizer.encode(word)[0][1:-1]

            if r < 0.15 * 0.8:
                source.extend(len(word_token) * [tokenizer._token_mask_id])
                target.extend(word_token)
            elif r < 0.15 * 0.9:
                source.extend(word_token)
                target.extend(word_token)
            elif r < 0.15:
                source.extend([np.random.choice(tokenizer._vocab_size - 5) + 5 for _ in range(len(word_token))])
                target.extend(word_token)
            else:
                source.extend(word_token)
                target.extend([0] * len(word_token))

        # add end token
        source.append(tokenizer._token_end_id)
        #         target.append(tokenizer._token_end_id) # if mask end token, use this line
        target.append(0)

        if i == 0:
            segment = [0] * len(source)
        else:
            segment = [1] * len(source)

        sources.extend(source)
        targets.extend(target)
        segments.extend(segment)

    return sources, targets, segments


class data_generator(DataGenerator):
    def __iter__(self, shuffle=False):
        batch_token_ids, batch_segment_ids, batch_target_ids, batch_is_masked, batch_nsp = [], [], [], [], []

        for is_end, item in self.get_sample(shuffle):
            # 50% shuffle order
            label = 1
            p = np.random.random()
            if p < 0.5:
                label = 0
                item = shuffle_reply(item)

            source_tokens, target_tokens, segment_ids = random_masking(item)

            is_masked = [0 if i == 0 else 1 for i in target_tokens]
            batch_token_ids.append(source_tokens)
            batch_segment_ids.append(segment_ids)
            batch_target_ids.append(target_tokens)
            batch_is_masked.append(is_masked)
            batch_nsp.append([label])

            if is_end or len(batch_token_ids) == self.batch_size:
                batch_token_ids = pad_sequences(batch_token_ids, maxlen=maxlen)
                batch_segment_ids = pad_sequences(batch_segment_ids, maxlen=maxlen)
                batch_target_ids = pad_sequences(batch_target_ids, maxlen=maxlen)
                batch_is_masked = pad_sequences(batch_is_masked, maxlen=maxlen)
                batch_nsp = pad_sequences(batch_nsp)

                yield [batch_token_ids, batch_segment_ids, batch_target_ids, batch_is_masked, batch_nsp], None

                batch_token_ids, batch_segment_ids, batch_target_ids, batch_is_masked = [], [], [], []
                batch_nsp = []


train_generator = data_generator(words_data, batch_size)


def build_transformer_model_with_mlm():
    """带mlm的bert模型
    """
    bert = build_transformer_model(
        config_path,
        with_mlm='linear',
        with_nsp=True,
        model='nezha',
        return_keras_model=False,
    )
    proba = bert.model.output
    #     print(proba)
    # 辅助输入
    token_ids = Input(shape=(None,), dtype='int64', name='token_ids')  # 目标id
    is_masked = Input(shape=(None,), dtype=K.floatx(), name='is_masked')  # mask标记
    nsp_label = Input(shape=(None,), dtype='int64', name='nsp')  # nsp

    def mlm_loss(inputs):
        """计算loss的函数，需要封装为一个层
        """
        y_true, y_pred, mask = inputs
        _, y_pred = y_pred
        loss = K.sparse_categorical_crossentropy(
            y_true, y_pred, from_logits=True
        )
        loss = K.sum(loss * mask) / (K.sum(mask) + K.epsilon())
        return loss

    def nsp_loss(inputs):
        """计算nsp loss的函数，需要封装为一个层
        """
        y_true, y_pred = inputs
        y_pred, _ = y_pred
        loss = K.sparse_categorical_crossentropy(
            y_true, y_pred
        )
        loss = K.mean(loss)
        return loss

    def mlm_acc(inputs):
        """计算准确率的函数，需要封装为一个层
        """
        y_true, y_pred, mask = inputs
        _, y_pred = y_pred
        y_true = K.cast(y_true, K.floatx())
        acc = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        acc = K.sum(acc * mask) / (K.sum(mask) + K.epsilon())
        return acc

    def nsp_acc(inputs):
        """计算准确率的函数，需要封装为一个层
        """
        y_true, y_pred = inputs
        y_pred, _ = y_pred
        y_true = K.cast(y_true, K.floatx())
        acc = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        acc = K.mean(acc)
        return acc

    mlm_loss = Lambda(mlm_loss, name='mlm_loss')([token_ids, proba, is_masked])
    mlm_acc = Lambda(mlm_acc, name='mlm_acc')([token_ids, proba, is_masked])
    nsp_loss = Lambda(nsp_loss, name='nsp_loss')([nsp_label, proba])
    nsp_acc = Lambda(nsp_acc, name='nsp_acc')([nsp_label, proba])

    train_model = Model(
        bert.model.inputs + [token_ids, is_masked, nsp_label], [mlm_loss, mlm_acc, nsp_loss, nsp_acc]
    )

    loss = {
        'mlm_loss': lambda y_true, y_pred: y_pred,
        'mlm_acc': lambda y_true, y_pred: K.stop_gradient(y_pred),
        'nsp_loss': lambda y_true, y_pred: y_pred,
        'nsp_acc': lambda y_true, y_pred: K.stop_gradient(y_pred),
    }

    return bert, train_model, loss


bert, train_model, loss = build_transformer_model_with_mlm()

Opt = extend_with_weight_decay(Adam)
Opt = extend_with_gradient_accumulation(Opt)
Opt = extend_with_piecewise_linear_lr(Opt)

opt = Opt(learning_rate=learning_rate,
          exclude_from_weight_decay=['Norm', 'bias'],
          lr_schedule={int(len(train_generator) * epochs * 0.1): 1.0, len(train_generator) * epochs: 0},
          weight_decay_rate=0.01,
          grad_accum_steps=2,
          )

train_model.compile(loss=loss, optimizer=opt)
# 如果传入权重，则加载。注：须在此处加载，才保证不报错。
if checkpoint_path is not None:
    bert.load_weights_from_checkpoint(checkpoint_path)

train_model.summary()


class ModelCheckpoint(keras.callbacks.Callback):
    """
        每10个epoch保存一次模型
    """

    def __init__(self):
        self.loss = 1e6

    def on_epoch_end(self, epoch, logs=None):
        if logs['loss'] < self.loss:
            self.loss = logs['loss']

        #         print('epoch: {}, loss is : {}, lowest loss is:'.format(epoch, logs['loss'], self.loss))

        if (epoch + 1) % 10 == 0:
            bert.save_weights_as_checkpoint(model_saved_path + '-{}'.format(epoch + 1))

        token_ids, segment_ids = tokenizer.encode(u'看哪个？', '微信您通过一下吧')
        token_ids[9] = token_ids[10] = tokenizer._token_mask_id

        probs = bert.model.predict([np.array([token_ids]), np.array([segment_ids])])[1]
        print(tokenizer.decode(probs[0, 9:11].argmax(axis=1)))


if __name__ == '__main__':
    # 保存模型
    checkpoint = ModelCheckpoint()
    # 记录日志
    csv_logger = keras.callbacks.CSVLogger('training.log')

    train_model.fit(
        train_generator.generator(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[checkpoint, csv_logger],
    )
