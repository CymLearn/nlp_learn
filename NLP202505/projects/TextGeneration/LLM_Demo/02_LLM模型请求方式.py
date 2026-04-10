# -*- coding: utf-8 -*-
"""
Create Date Time : 2026/3/14 16:37
Create User : 19410
Desc : 调用OpenAI格式的模型接口
"""


def invoke_with_http():
    """
    通过http请求的方式来调用接口
    :return:
    """
    import requests

    url = "http://8.136.193.178:20000"
    api_key = None
    model_name = "Qwen1___5-0___5B-Chat"
    model_name = "xinghuo-api"

    url = 'https://api.deepseek.com'
    api_key = 'sk-ea81311cf8574a1d81c60a9f5f6fc059'  # 这个会删除的
    model_name = "deepseek-chat"

    # 一个免费的大模型的网站: https://openrouter.ai/
    url = 'https://openrouter.ai/api'
    api_key = "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"
    model_name = "qwen/qwen3-235b-a22b-2507"

    # OpenAI API转发 cloudflare
    url = "https://gateway.ai.cloudflare.com/v1/67b8ebfcb6b836e009e1fb540f160fa5/nlp_0314/deepseek"
    api_key = 'sk-ea81311cf8574a1d81c60a9f5f6fc059'  # 这个会删除的
    model_name = "deepseek-chat"

    # OpenAI API转发 cloudflare
    url = "https://gateway.ai.cloudflare.com/v1/67b8ebfcb6b836e009e1fb540f160fa5/nlp_0314/openrouter"
    api_key = "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"
    model_name = "qwen/qwen3-235b-a22b-2507"

    headers = {
        "accept": "application/json",
        "Content-Type": "application/json"
    }
    if api_key is not None:
        headers['Authorization'] = f'Bearer {api_key}'

    def _list_models():
        try:
            _url = f"{url}/v1/models"
            _response = requests.get(_url, headers=headers)
            if _response.status_code == 200:
                print(f"list_models结果为: {_response.json()}")
            else:
                print(f"list_models异常:{_response.status_code}")
        except Exception as e:
            print(f"获取模型列表异常:{e}")

    def _chat_with_llm(_model_name=None):
        _model_name = _model_name or "Qwen1___5-0___5B-Chat"
        _url = f"{url}/v1/chat/completions"

        _messages = []
        while True:
            msg = input("我:")
            if msg == 'q':
                break
            _messages.append({'role': 'user', 'content': msg})
            if len(_messages) > 5:
                # 针对过去的历史信息仅保留部分 --> 保留最近3轮的
                _messages = _messages[-5:]

            _data = {
                "model": _model_name,
                "messages": _messages,
                "temperature": 0.7,
                "stream": False  # 非流式返回结果
            }
            _response = requests.post(
                _url, json=_data, headers=headers
            )
            if _response.status_code == 200:
                _result = _response.json()
                # print(f"完成返回的数据为:{_result}")
                _ai_role = _result['choices'][-1]['message']['role']
                _ai_msg = _result['choices'][-1]['message']['content']
                print(f"chat结果: {_ai_msg}")
                _messages.append({'role': _ai_role, 'content': _ai_msg})
            else:
                print(f"chat结果异常:{_response.status_code}")
                _messages = _messages[:-1]

    _list_models()
    _chat_with_llm(model_name)


if __name__ == '__main__':
    invoke_with_http()
