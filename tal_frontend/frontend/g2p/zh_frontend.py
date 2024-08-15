# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import re
from typing import List

import jieba.posseg as psg
import numpy as np
#from g2pM import G2pM
import time
from pypinyin import lazy_pinyin
from pypinyin import load_phrases_dict
from pypinyin import load_single_dict
from pypinyin import Style
from pypinyin_dict.phrase_pinyin_data import large_pinyin

#from tal_frontend.frontend.g2pw import G2PWOnnxConverter
#from tal_frontend.frontend.generate_lexicon import generate_lexicon
#from tal_frontend.frontend.polyphonic import Polyphonic

INITIALS = [
    'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'zh', 'ch', 'sh',
    'r', 'z', 'c', 's', 'j', 'q', 'x'
]
INITIALS += ['y', 'w', 'sp', 'spl', 'spn', 'sil']

# 0 for None, 5 for neutral
TONES = ["0", "1", "2", "3", "4", "5"]

def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result

def insert_after_character(lst, item):
    """
    inset `item` after finals.
    """
    result = [item]

    for phone in lst:
        result.append(phone)
        if phone not in INITIALS:
            # finals has tones
            # assert phone[-1] in "12345"
            result.append(item)

    return result


class Frontend():
    def __init__(self,
                 phone_vocab_path=None,
                 tone_vocab_path=None,
                 ):

        self.punc = "、：，；。？！“”‘’':,;.?!"
        self.rhy_phns = ['sp1', 'sp2', 'sp3', 'sp4']
        self.phrases_dict = {
            '开户行': [['ka1i'], ['hu4'], ['hang2']],
            '发卡行': [['fa4'], ['ka3'], ['hang2']],
            '放款行': [['fa4ng'], ['kua3n'], ['hang2']],
            '茧行': [['jia3n'], ['hang2']],
            '行号': [['hang2'], ['ha4o']],
            '各地': [['ge4'], ['di4']],
            '借还款': [['jie4'], ['hua2n'], ['kua3n']],
            '时间为': [['shi2'], ['jia1n'], ['we2i']],
            '为准': [['we2i'], ['zhu3n']],
            '色差': [['se4'], ['cha1']],
            '嗲': [['dia3']],
            '呗': [['bei5']],
            '不': [['bu4']],
            '咗': [['zuo5']],
            '嘞': [['lei5']],
            '掺和': [['chan1'], ['huo5']],
            '三行':[['san1'],['hang2']]
            #'嗯':[['en4']]
        }

        self.must_erhua = {
            "小院儿", "胡同儿", "范儿", "老汉儿", "撒欢儿", "寻老礼儿", "妥妥儿", "媳妇儿", "红包儿", "男儿",
        }
        self.not_erhua = {
            "虐儿", "为儿", "护儿", "瞒儿", "救儿", "替儿", "有儿", "一儿", "我儿", "俺儿", "妻儿",
            "拐儿", "聋儿", "乞儿", "患儿", "幼儿", "孤儿", "婴儿", "婴幼儿", "连体儿", "脑瘫儿",
            "流浪儿", "体弱儿", "混血儿", "蜜雪儿", "舫儿", "祖儿", "美儿", "应采儿", "可儿", "侄儿",
            "孙儿", "侄孙儿", "女儿", "男儿", "红孩儿", "花儿", "虫儿", "马儿", "鸟儿", "猪儿", "猫儿",
            "狗儿", "少儿"
        }

        self.vocab_phones = {}
        self.vocab_tones = {}
        if phone_vocab_path:
            with open(phone_vocab_path, 'rt', encoding='utf-8') as f:
                phn_id = [line.strip().split() for line in f.readlines()]
            for phn, id in phn_id:
                self.vocab_phones[phn] = int(id)
        if tone_vocab_path:
            with open(tone_vocab_path, 'rt', encoding='utf-8') as f:
                tone_id = [line.strip().split() for line in f.readlines()]
            for tone, id in tone_id:
                self.vocab_tones[tone] = int(id)

        # g2p
        self._init_pypinyin()

    def _init_pypinyin(self):
        """
        Load pypinyin G2P module.
        """
        large_pinyin.load()
        load_phrases_dict(self.phrases_dict)
        # 调整字的拼音顺序
        load_single_dict({ord(u'地'): u'de,di4'})

    def _get_initials_finals(self, word: str) -> List[List[str]]:
        """
        Get word initial and final by pypinyin or g2pM
        """
        initials = []
        finals = []
        orig_initials = lazy_pinyin(
            word, neutral_tone_with_five=True, style=Style.INITIALS)
        orig_finals = lazy_pinyin(
            word, neutral_tone_with_five=True, style=Style.FINALS_TONE3)
        for c, v in zip(orig_initials, orig_finals):
            if re.match(r'i\d', v):
                if c in ['z', 'c', 's']:
                    # zi, ci, si
                    v = re.sub('i', 'ii', v)
                elif c in ['zh', 'ch', 'sh', 'r']:
                    # zhi, chi, shi
                    v = re.sub('i', 'iii', v)
            #修改n为en
            #elif re.match(r'n\d', v):
            #        v = re.sub('n','en', v)
            initials.append(c)
            finals.append(v)

        return initials, finals

    def _merge_erhua(self,
                     initials: List[str],
                     finals: List[str],
                     word: str,
                     pos: str) -> List[List[str]]:
        """
        Do erhub.
        """
        # fix er1
        for i, phn in enumerate(finals):
            if i == len(finals) - 1 and word[i] == "儿" and phn == 'er1':
                finals[i] = 'er2'

        # 发音
        if word not in self.must_erhua and (word in self.not_erhua or
                                            pos in {"a", "j", "nr"}):
            return initials, finals

        # "……" 等情况直接返回
        if len(finals) != len(word):
            return initials, finals

        assert len(finals) == len(word)

        # 不发音
        new_initials = []
        new_finals = []
        for i, phn in enumerate(finals):
            if i == len(finals) - 1 and word[i] == "儿" and phn in {
                    "er2", "er5"
            } and word[-2:] not in self.not_erhua and new_finals:
                # new_finals[-1] = new_finals[-1][:-1] + "r" + new_finals[-1][-1]
                new_finals.append('er5')           #这里儿化音改为前面加上x
                new_initials.append('x')
            else:
                new_initials.append(initials[i])
                new_finals.append(phn)

        return new_initials, new_finals

    # if merge_sentences, merge all sentences into one phone sequence
    def _g2p(self,
             sentences: List[str],
             merge_sentences: bool=True,
             with_erhua: bool=True) -> List[List[str]]:
        """
        Return: list of list phonemes.
            [['w', 'o3', 'm', 'en2', 'sp'], ...]
        """
        segments = sentences
        phones_list = []

        # split by punctuation
        for seg in segments:
            # remove all English words in the sentence
            seg = re.sub('[a-zA-Z]+', '', seg)
            # [(word, pos), ...]
            seg_cut = psg.lcut(seg)
            # fix wordseg bad case for sandhi
            #seg_cut = self.tone_modifier.pre_merge_for_modify(seg_cut)

            # 为了多音词获得更好的效果，这里采用整句预测
            phones = []
            initials = []
            finals = []

            for word, pos in seg_cut:
                if pos == 'eng':
                    continue
                sub_initials, sub_finals = self._get_initials_finals(word)

                if with_erhua:
                    sub_initials, sub_finals = self._merge_erhua(
                        sub_initials, sub_finals, word, pos)

                initials.append(sub_initials)
                finals.append(sub_finals)

            initials = sum(initials, [])
            finals = sum(finals, [])

            for c, v in zip(initials, finals):
                # NOTE: post process for pypinyin outputs
                # we discriminate i, ii and iii
                if c and c not in self.punc:
                    phones.append(c)
                # replace punctuation by `sp`
                if c and c in self.punc:
                    phones.append('sp')

                if v and v not in self.punc and v not in self.rhy_phns:
                    phones.append(v)

            phones_list.append(phones)

        # merge split sub sentence into one sentence.
        if merge_sentences:
            # sub sentence phonemes
            merge_list = sum(phones_list, [])
            # rm the last 'sp' to avoid the noise at the end
            # cause in the training data, no 'sp' in the end
            if merge_list[-1] == 'sp':
                merge_list = merge_list[:-1]

            # sentence phonemes
            phones_list = []
            phones_list.append(merge_list)

        return phones_list


    def get_phonemes(self,
                     sentence: str,
                     merge_sentences: bool=True,
                     with_erhua: bool=True,
                     robot: bool=False,
                     ) -> List[List[str]]:
        """
        Main function to do G2P
        """
        # TN & Text Segmentation
        sentences = sentence.strip().split()
        # Prosody & WS & g2p & tone sandhi
        phonemes = self._g2p(
            sentences, merge_sentences=merge_sentences, with_erhua=with_erhua)
        # simulate robot pronunciation, change all tones to `1`
        if robot:
            new_phonemes = []
            for sentence in phonemes:
                new_sentence = []
                for item in sentence:
                    # `er` only have tone `2`
                    if item[-1] in "12345" and item != "er2":
                        item = item[:-1] + "1"
                    new_sentence.append(item)
                new_phonemes.append(new_sentence)
            phonemes = new_phonemes

        return ' '.join(phonemes[0])
        #return phonemes

