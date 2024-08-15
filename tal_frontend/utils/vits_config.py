__all__ = ['VALID_PARAMS', 'SPK_ID', 'DEFAULT_SPEAK_INFO', 'VITS_MODEL', 'DEFAULT_FORMULATE_PARAMS']


# 定义合法值集合和范围
VALID_PARAMS = {
    'voice_type': {'values': {'xiaosi'}, 'range': None},
    'emotion': {'values': {'neutral', 'happy', 'happiness','sadness', 'sad', 'angry', 'confusion', 'fear', 'disgruntled', 'cheerful'}, 'range': None},
    'audio_format': {'values': {'mp3', 'wav', 'pcm'}, 'range': None},
    'lang': {'values': {'cn', 'en'}, 'range': None},
    'rate': {'values': None, 'range': (0.5, 2)},
    'pitch': {'values': None, 'range': (0.5, 1.5)},
    'volume': {'values': None, 'range': (0, 100)}
}

# 说话人及情感映射字典
SPK_ID = {
    'xiaosi_cn_neutral': '1',
    'xiaosi_cn_confusion': '286',
    'xiaosi_cn_happy': '283',
    'xiaosi_cn_happiness': '283',
    'xiaosi_cn_sad': '282',
    'xiaosi_cn_sadness': '282',
    'xiaosi_cn_angry': '285',
    'xiaosi_cn_fear': '284',
    'xiaosi_cn_disgruntled': '282',
    'xiaosi_cn_cheerful': '284',
    # 'xiaosi_en_neutral': '6',
    'xiaosi_en_neutral': '1',
}

VITS_MODEL = {
    "xiaosi_cn_neutral": "tal_vits",
    "xiaosi_cn_confusion": "tal_vits",
    "xiaosi_cn_happy": "tal_vits",
    "xiaosi_cn_happiness": "tal_vits",
    "xiaosi_cn_sad": "tal_vits_fear_sad",
    "xiaosi_cn_sadness": "tal_vits_fear_sad",
    "xiaosi_cn_fear": "tal_vits_fear_sad",
    "xiaosi_cn_angry": "tal_vits_angry",
    "xiaosi_cn_disgruntled": "tal_vits_cheerful_disgruntled",
    "xiaosi_cn_cheerful": "tal_vits_cheerful_disgruntled",
    # 英文发音有待提升，暂用中文模型
    # "xiaosi_en_neutral": "tal_vits_english",
    "xiaosi_en_neutral": "tal_vits",
}

DEFAULT_SPEAK_INFO = {
    'voice_type': 'xiaosi',
    'emotion': 'neutral',
    'lang': 'cn',
    'audio_format': 'wav',
    'rate': '1',
    'pitch': '1',
    'volume': '100'
}

DEFAULT_FORMULATE_PARAMS = {
    'space_norm': False,
    'graphic_type': None
}
