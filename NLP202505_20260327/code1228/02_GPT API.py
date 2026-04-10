# -*- coding: utf-8 -*-
"""
Create Date Time : 2025/12/28 11:59
Create User : 19410
Desc : xxx

https://modelscope.cn/models/openai-community/gpt2/files
https://hf-mirror.com/openai-community/gpt2
"""

import os

import torch

# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['XDG_CACHE_HOME'] = r"D:\huggingface"
os.environ['CACHE_HOME'] = r'D:\huggingface'


def download_model():
    from transformers import AutoModel, AutoTokenizer, GPT2TokenizerFast, GPT2Model

    # Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. Falling back to regular HTTP download. For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`

    model_id = "openai-community/gpt2"
    # 分词器的恢复加载
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # 模型恢复
    model = AutoModel.from_pretrained(model_id)
    print(type(tokenizer))
    print(type(model))
    print(model)



def use_gpt_pipeline():
    from transformers import pipeline, set_seed
    from transformers.pipelines.text_generation import TextGenerationPipeline

    path = r"openai-community/gpt2"
    print(f"模型加载路径:{path}")
    """
    pipeline: 将所有操作合并到一起，达到一个效果：给定一个输入，得到一个明确的输出
        eg: 给定了文本前缀的，得到生成好的完整文本数据
    """
    pipe = pipeline(
        'text-generation',
        model=path,
        framework="pt"
    )
    print(type(pipe))
    print(type(pipe.model))
    print(type(pipe.tokenizer))

    set_seed(42)

    # r = pipe("Hello, I'm a language model,", max_length=30, num_return_sequences=5)
    r = pipe("我是中国人，我比较喜欢吃", max_length=100, num_return_sequences=5)
    print(r)


def use_gpt_model():
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    path = r"D:\huggingface\huggingface\hub\models--openai-community--gpt2\snapshots\607a30d783dfa663caf39e06633721c8d4cfcd7e"
    print(f"模型加载路径:{path}")

    # GPT中的分词采用的时候BBPE(Byte-Level Byte-Pair Encoding)
    tokenizer = GPT2Tokenizer.from_pretrained(path)
    model = GPT2LMHeadModel.from_pretrained(path)

    text = "我是中国"
    # text = "a b c d e f g h i"
    # 分词+词id转换 [1,6] eg:[[22755,   239, 42468, 40792, 32368,   121]]
    input_ids = tokenizer(text, return_tensors='pt')['input_ids']
    # 获取gpt模型的输出
    gpt_output = model(input_ids)
    # 每个样本的每个token预测属于C个类别的置信度 [bs,t,vocab_size]
    pred_token_scores = gpt_output.logits  # [1,6,50257]
    pred_token_ids = torch.argmax(pred_token_scores, dim=-1) # [1,6,50257] --> [1,6]

    # 类别id还原token文本
    text = tokenizer.decode(
        pred_token_ids[0][-1:], # 最后一个预测token
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    print(text)

if __name__ == '__main__':
    # download_model()
    # use_gpt_model()
    use_gpt_pipeline()
