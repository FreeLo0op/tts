import json
import requests
import itertools
from tal_frontend.frontend.g2p.utils import ph2id, rhy2id

class RequestFailed(Exception):
    """自定义异常类，用于处理无效参数错误"""
    pass

def llm_request(text: str,url: str):
    headers = {
        'Content-Type': 'application/json'
    }

    data = {
        'user_id': '',
        'request_id': '974d59c5-34b8-4635-a797-0fe4f3c030a0',
        'session_id': '',
        'trace_id': '',
        'span_id': '',
        'text': text
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        res = json.loads(response.text)
        phonemes, prosodies = res['phonemes'], res['prosodies']
        phonemes = list(itertools.chain(*phonemes))
        prosodies = list(itertools.chain(*prosodies))
        # 删除xer
        # 替换xer为 er2
        indices = sorted([index for index, value in enumerate(phonemes) if value == 'xer'], reverse=True)
        for index in indices:
            phonemes[index] = 'er2'
        #    del phonemes[index]
        #    del prosodies[index]

        #return phonemes, prosodies
        phonemes_id = ph2id(phonemes)
        prosodies_id = rhy2id(prosodies)
        if len(prosodies_id) == len(phonemes_id) + 1:
            prosodies_id = prosodies_id[:-1]
            prosodies_id[-1] = 1
            prosodies = prosodies[:-1]
        return phonemes_id, prosodies_id, phonemes, prosodies
        return response.text
    else:
        raise RequestFailed(f'韵律和g2p请求失败, 状态码:{response.status_code}')
    
def math_request(text: str,url:str, type: str=None, space_norm:bool=False, graphic_type:str=None):
    # 设置请求的URL
    # 设置请求的数据和头部
    headers = {'Content-Type': 'application/json'}
    data = {
        'text': text,
        'type': type,
        'trace_ud': '123456'
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        res = json.loads(response.text)
        res = res['results']# + '，'
        return res 
    else:
        raise RequestFailed(f'数学公式解析请求失败, 状态码:{response.status_code}')