# -*- coding: utf-8 -*-
# @Date    : 2020/12/14
# @Author  : mingming.xu
# @Email   : xv44586@gmail.com
# @File    : pair-self-kd.py
import os
import numpy as np
import tensorflow as tf
import random
import numpy as np
from tqdm import tqdm

from toolkit4nlp.utils import *
from toolkit4nlp.models import *
from toolkit4nlp.tokenizers import *
from toolkit4nlp.backend import *
from toolkit4nlp.layers import *
from toolkit4nlp.optimizers import *

seed = 0
# tf.random.set_seed(seed)
np.random.seed(seed)

path = '/home/mingming.xu/datasets/NLP/ccf_qa_match/'

maxlen = 128
batch_size = 16
epochs = 5
Temperature = 4  # 平滑soften labels 分布，越大越平滑，一般取值[1, 10]

# bert配置
config_path = '/home/mingming.xu/pretrain/NLP/nezha_base_wwm/bert_config.json'
checkpoint_path = './nezha_post_training/wwm-model-add-dict-no-mask-end-sop.ckpt-40'
dict_path = '/home/mingming.xu/pretrain/NLP/nezha_base_wwm/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


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


train_data_all = load_data('train')
test_data = load_data('test')


# data generator
def can_padding(token_id):
    if token_id in (tokenizer._token_mask_id, tokenizer._token_end_id, tokenizer._token_start_id):
        return False
    return True


class data_generator(DataGenerator):
    def random_padding(self, token_ids):
        rands = np.random.random(len(token_ids))
        new_tokens = []
        for p, token in zip(rands, token_ids):
            if p < 0.1 and can_padding(token):
                new_tokens.append(tokenizer._token_pad_id)
            else:
                new_tokens.append(token)
        return new_tokens

    def __iter__(self, shuffle=False):
        batch_token_ids, batch_segment_ids, batch_labels, batch_soften = [], [], [], []
        for is_end, item in self.get_sample(shuffle):
            soften = None
            if len(item) == 5:
                q_id, q, r_id, r, label = item
            else:
                # has soften
                q_id, q, r_id, r, label, soften = item

            label = int(label) if label is not None else None

            token_ids, segment_ids = tokenizer.encode(q, r, maxlen=256)
            if shuffle:
                token_ids = self.random_padding(token_ids)

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            batch_soften.append(soften)

            if is_end or len(batch_token_ids) == self.batch_size:
                batch_token_ids = pad_sequences(batch_token_ids)
                batch_segment_ids = pad_sequences(batch_segment_ids)
                batch_labels = pad_sequences(batch_labels)
                if soften is not None:
                    batch_soften = pad_sequences(batch_soften)
                if len(item) == 5:
                    yield [batch_token_ids, batch_segment_ids], batch_labels
                else:
                    yield [batch_token_ids, batch_segment_ids], [batch_labels, batch_soften]

                batch_token_ids, batch_segment_ids, batch_labels, batch_soften = [], [], [], []


# shuffle
# np.random.shuffle(train_data)
# n = int(len(train_data) * 0.8)
# train_generator = data_generator(train_data[:n], batch_size)
# valid_generator = data_generator(train_data[n: ], batch_size)
# test_generator = data_generator(test_data, batch_size)


fold = 0
train_data, valid_data = [], []
for idx in range(len(train_data_all)):
    if int(train_data_all[idx][0]) % 10 != fold:
        train_data.append(train_data_all[idx])
    else:
        valid_data.append(train_data_all[idx])

train_generator = data_generator(train_data, batch_size=batch_size)
valid_generator = data_generator(valid_data, batch_size=batch_size * 2)
test_generator = data_generator(test_data, batch_size=batch_size * 2)

# 构建Teacher model
teacher = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='nezha',
    prefix='Teacher-'
)

output = Lambda(lambda x: x[:, 0])(teacher.output)
output = Dropout(0.01)(output)
logits = Dense(2)(output)
t_output = Activation(activation='softmax')(logits)
teacher_model = keras.models.Model(teacher.input, t_output)
teacher_logits = keras.models.Model(teacher.input, logits)
teacher_model.summary()

grad_accum_steps = 3
opt = extend_with_weight_decay(Adam)
opt = extend_with_gradient_accumulation(opt)
exclude_from_weight_decay = ['Norm', 'bias']
opt = extend_with_piecewise_linear_lr(opt)
para = {
    'learning_rate': 1e-5,
    'weight_decay_rate': 0.1,
    'exclude_from_weight_decay': exclude_from_weight_decay,
    'grad_accum_steps': grad_accum_steps,
    'lr_schedule': {int(len(train_generator) * 0.1 * epochs / grad_accum_steps): 1,
                    int(len(train_generator) * epochs / grad_accum_steps): 0},
}

opt = opt(**para)
teacher_model.compile(
    loss='sparse_categorical_crossentropy',
    #     optimizer=Adam(2e-5),  # 用足够小的学习率
    optimizer=opt,
    metrics=['accuracy'],
)


def evaluate(data=valid_generator, model=None):
    P, R, TP = 0., 0., 0.
    for x_true, y_true in tqdm(data):
        y_pred = model.predict(x_true).argmax(-1)
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

    def __init__(self, save_path, valid_model=None):
        self.best_val_f1 = 0.
        self.save_path = save_path
        self.valid_model = valid_model

    def on_epoch_end(self, epoch, logs=None):
        val_f1 = evaluate(valid_generator, self.valid_model)
        if val_f1 > self.best_val_f1:
            self.best_val_f1 = val_f1
            self.model.save_weights(self.save_path)
        #         test_acc = evaluate(test_generator)
        print(
                u'val_f1: %.5f, best_val_f1: %.5f\n' %
                (val_f1, self.best_val_f1)
        )


# build student model
student = build_transformer_model(config_path=config_path,
                                  checkpoint_path=checkpoint_path,
                                  model='nezha',
                                  prefix='Student-')

x = Lambda(lambda x: x[:, 0])(student.output)
x = Dropout(0.01)(x)
s_logits = Dense(2)(x)
s_output = Activation(activation='softmax')(s_logits)

student_model = Model(student.input, s_output)

s_logits_t = Lambda(lambda x: x / Temperature)(s_logits)
s_soften = Activation(activation='softmax')(s_logits_t)

student_train = Model(student.inputs, [s_output, s_soften])

student_train.summary()
student_train.compile(
    loss=['sparse_categorical_crossentropy', keras.losses.kullback_leibler_divergence],
    optimizer=opt,
    loss_weights=[1, Temperature ** 2])  # 放大kld

if __name__ == '__main__':

    teacher_save_path = 'pair-teacher.model'

    evaluator = Evaluator(save_path=teacher_save_path, valid_model=teacher_model)

    teacher_model.fit_generator(
        train_generator.generator(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator],
    )
    teacher_model.load_weights(teacher_save_path)
    evaluate(valid_generator, teacher_model)

    # create  logits
    logits = []
    for x, _ in tqdm(data_generator(train_data_all, 64)):
        logit_ = teacher_logits.predict(x)
        logits.append(logit_)

    logits = np.concatenate(logits, axis=0)
    logits.dump('train_all.logits')

    # logits = np.load('train_all.logits', allow_pickle=True)
    train_data_logits = []
    for d, l in zip(train_data_all, logits):
        soften = K.softmax(l / Temperature).numpy().tolist()
        train_data_logits.append(d + [soften])

    train_data = []
    for idx in range(len(train_data_logits)):
        if int(train_data_logits[idx][0]) % 10 != fold:
            train_data.append(train_data_logits[idx])

    student_train_generator = data_generator(train_data, batch_size=batch_size)

    student_save_path = 'pair-student.model'
    student_evaluator = Evaluator(student_save_path, student_model)

    student_train.fit_generator(student_train_generator.generator(),
                                steps_per_epoch=len(student_train_generator),
                                epochs=epochs,
                                callbacks=[student_evaluator])

    student_train.load_weights(student_save_path)
    evaluate(valid_generator, student_model)
