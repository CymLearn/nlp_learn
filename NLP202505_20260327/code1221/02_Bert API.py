# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/12/21 16:21
Create User : 19410
Desc : Bert模型的使用Demo

NOTE:
    默认情况下，会下载到当前用户根目录下的.cache/huggingface文件夹中
    但是可以通过给定环境变量:
        XDG_CACHE_HOME=xxx 来指定模型保存文件路径

# pip install transformers==4.57.3 -i https://mirrors.aliyun.com/pypi/simple
# 由于transformers框架对应的网站 https://huggingface.co 需要外网访问，所以这里弄一个国内使用的网站
# 国内网站：https://hf-mirror.com   --- 有做访问限制的处理

"""

import os

# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['XDG_CACHE_HOME'] = r"D:\cache"
os.environ['CACHE_HOME'] = r'D:\cache'


def tt_v0():
    from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel, BertModel

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    print(f"Bert对应的分词器:\n{type(tokenizer)}\n{tokenizer}\n")
    model = BertModel.from_pretrained("bert-base-chinese")
    print(f"Bert对应的模型:\n{type(model)}\n{model}\n")


def tt_v1():
    from transformers import pipeline

    unmasker = pipeline('fill-mask', model='bert-base-chinese')
    print("=" * 50)
    print(unmasker("The man worked as a [MASK]."))
    print("=" * 50)
    print(unmasker("The woman worked as a [MASK]."))
    print("=" * 50)
    print(unmasker("中国的首都是[MASK]京。"))


def tt_v2():
    from transformers import BertTokenizer, BertModel, BertConfig
    from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertModel.from_pretrained("bert-base-chinese")
    text = "还有双鸭山到淮阴的汽车票吗13号的"
    # 基于给定文本得到文本对应的分词结果以及token id转换结果
    # encoded_input = tokenizer(text, return_tensors='pt')
    encoded_input = tokenizer([text, "从这里怎么回家"], return_tensors='pt', padding=True)
    # 将token id输入模型得到模型输出：默认仅包括bert最后一层的输出特征向量
    output = model(**encoded_input)
    print(type(output))
    # 获取bert模型最后一层的输出特征向量 [bs,t,hidden_size]
    last_hidden_state = output[0]  # output.last_hidden_state
    print(last_hidden_state.shape)
    cls_hidden_state = last_hidden_state[:, 0, :]  # 获取CLS这个token对应的特征向量
    print(cls_hidden_state.shape)


if __name__ == '__main__':
    tt_v2()
