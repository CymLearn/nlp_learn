#! -*- coding:utf-8 -*-
# bert+crf用来做实体识别
# 数据集：http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz
# [valid_f1]  token_level: 97.06； entity_level: 95.90
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from bert4torch.callbacks import Callback
from bert4torch.snippets import sequence_padding, ListDataset, seed_everything
from bert4torch.layers import CRF
from bert4torch.tokenizers import Tokenizer
from bert4torch.models import build_transformer_model, BaseModel
from tqdm import tqdm

maxlen = 256
batch_size = 16
categories = [
    'O', 'B-实验室检验', 'I-实验室检验', 'B-影像检查', 'I-影像检查', 'B-手术', 'I-手术',
    'B-疾病和诊断', 'I-疾病和诊断', 'B-药物', 'I-药物', 'B-解剖部位', 'I-解剖部位'
]
categories_id2label = {i: k for i, k in enumerate(categories)}
categories_label2id = {k: i for i, k in enumerate(categories)}

# BERT base
model_dir = r"D:\cache\models--bert-base-chinese"
config_path = f'{model_dir}/bert4torch_config.json'
checkpoint_path = f'{model_dir}/pytorch_model.bin'
dict_path = f'{model_dir}/vocab.txt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 固定seed
seed_everything(42)


# 加载数据集
class MyDataset(ListDataset):
    @staticmethod
    def load_data(filename):
        D = []

        with open(filename, "r", encoding="utf-8") as reader:
            for line in reader:  # 遍历文件中的每一行数据
                line = line.strip()  # 前后空格及不可见字符去除
                obj = json.loads(line)  # json字符串转换为obj对象(字典)

                # 获取得到当前数据的原始文本
                text = obj['originalText']
                d = [text]

                # 对应的token 类别id
                for entity in obj['entities']:
                    label_type = entity['label_type']  # 实体类别名称
                    start_pos = entity['start_pos']  # 实体在token中起始位置，包含
                    end_pos = entity['end_pos']  # 实体在token中结束位置，不包含

                    d.append([start_pos, end_pos - 1, label_type])

                D.append(d)
        return D


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


def collate_fn(batch):
    batch_token_ids, batch_labels = [], []
    for d in batch:
        tokens = tokenizer.tokenize(d[0], maxlen=maxlen)
        mapping = tokenizer.rematch(d[0], tokens)
        start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
        end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
        token_ids = tokenizer.tokens_to_ids(tokens)
        labels = np.zeros(len(token_ids))
        for start, end, label in d[1:]:
            if start in start_mapping and end in end_mapping:
                start = start_mapping[start]
                end = end_mapping[end]
                labels[start] = categories_label2id['B-' + label]
                labels[start + 1:end + 1] = categories_label2id['I-' + label]
        batch_token_ids.append(token_ids)
        batch_labels.append(labels)
    batch_token_ids = torch.tensor(sequence_padding(batch_token_ids), dtype=torch.long, device=device)
    batch_labels = torch.tensor(sequence_padding(batch_labels), dtype=torch.long, device=device)
    return batch_token_ids, batch_labels


# 转换数据集
# noinspection PyTypeChecker
train_dataloader = DataLoader(
    MyDataset(r"../datas/medical/min_training.txt"),
    batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)
# noinspection PyTypeChecker
valid_dataloader = DataLoader(
    MyDataset(r"../datas/medical/min_val.txt"),
    batch_size=batch_size, collate_fn=collate_fn
)


# 定义bert上的模型结构
class Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.bert = build_transformer_model(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            segment_vocab_size=0
        )
        self.fc = nn.Linear(768, len(categories))  # 包含首尾

        # CRF ---> 需要在全连接(Softmax)的基础上额外的训练一个标签-标签的序列转换的置信度矩阵(转移概率矩阵/状态转移概率矩阵/类别标签转移概率矩阵)，
        # 也就是上一个时刻是A的时候，当前时刻是各个类别标签的置信度值
        # 最终当前时刻属于类别A的置信度是由两部分构成的：全连接(Softmax)输出属于类别A的置信度 + CRF中学习得到的上一个时刻类别X到当前时刻类别A的置信度
        self.crf = CRF(len(categories))

    def forward(self, token_ids):
        sequence_output = self.bert([token_ids])  # 得到bert的输出特征向量 [btz, seq_len, hdsz]
        emission_score = self.fc(sequence_output)  # 得到每个token属于各个类别的置信度 [btz, seq_len, tag_size]
        attention_mask = token_ids.gt(0).long()
        return emission_score, attention_mask

    def predict(self, token_ids):
        self.eval()
        with torch.no_grad():
            # 得到模型预测各个token属于各个类别的置信度信息以及mask矩阵信息
            emission_score, attention_mask = self.forward(token_ids)
            # 结合crf的模型参数，得到最终各个token对应的预测类别标签
            best_path = self.crf.decode(emission_score, attention_mask)  # [btz, seq_len]
            # best_path = torch.argmax(emission_score, dim=-1)
        return best_path


model = Model().to(device)


class Loss(nn.Module):
    def forward(self, outputs, labels):
        return model.crf(*outputs, labels)


def acc(y_pred, y_true):
    y_pred = y_pred[0]
    y_pred = torch.argmax(y_pred, dim=-1)
    acc = torch.sum(y_pred.eq(y_true)).item() / y_true.numel()
    return {'acc': acc}


# 支持多种自定义metrics = ['accuracy', acc, {acc: acc}]均可
model.compile(loss=Loss(), optimizer=optim.Adam(model.parameters(), lr=2e-5), metrics=acc)


def evaluate(data):
    X, Y, Z = 1e-10, 1e-10, 1e-10
    X2, Y2, Z2 = 1e-10, 1e-10, 1e-10
    for token_ids, label in tqdm(data):
        scores = model.predict(token_ids)  # [btz, seq_len]
        attention_mask = label.gt(0)

        # token粒度
        X += (scores.eq(label) * attention_mask).sum().item()
        Y += scores.gt(0).sum().item()
        Z += label.gt(0).sum().item()

        # entity粒度
        entity_pred = trans_entity2tuple(scores)
        entity_true = trans_entity2tuple(label)
        X2 += len(entity_pred.intersection(entity_true))
        Y2 += len(entity_pred)
        Z2 += len(entity_true)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    f2, precision2, recall2 = 2 * X2 / (Y2 + Z2), X2 / Y2, X2 / Z2
    return f1, precision, recall, f2, precision2, recall2


def trans_entity2tuple(scores):
    '''把tensor转为(样本id, start, end, 实体类型)的tuple用于计算指标
    '''
    batch_entity_ids = set()
    for i, one_samp in enumerate(scores):
        entity_ids = []
        for j, item in enumerate(one_samp):
            flag_tag = categories_id2label[item.item()]
            if flag_tag.startswith('B-'):  # B
                entity_ids.append([i, j, j, flag_tag[2:]])
            elif len(entity_ids) == 0:
                continue
            elif (len(entity_ids[-1]) > 0) and flag_tag.startswith('I-') and (flag_tag[2:] == entity_ids[-1][-1]):  # I
                entity_ids[-1][-2] = j
            elif len(entity_ids[-1]) > 0:
                entity_ids.append([])

        for i in entity_ids:
            if i:
                batch_entity_ids.add(tuple(i))
    return batch_entity_ids


class Evaluator(Callback):
    """评估与保存
    """

    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, steps, epoch, logs=None):
        f1, precision, recall, f2, precision2, recall2 = evaluate(valid_dataloader)
        if f2 > self.best_val_f1:
            self.best_val_f1 = f2
            # model.save_weights('best_model.pt')
        print(f'[val-token  level] f1: {f1:.5f}, p: {precision:.5f} r: {recall:.5f}')
        print(
            f'[val-entity level] f1: {f2:.5f}, p: {precision2:.5f} r: {recall2:.5f} best_f1: {self.best_val_f1:.5f}\n')


if __name__ == '__main__':

    evaluator = Evaluator()
    model.fit(train_dataloader, epochs=20, steps_per_epoch=None, callbacks=[evaluator])

else:

    model.load_weights('best_model.pt')
