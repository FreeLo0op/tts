# Copyright (c) 2022 Zhendong Peng (pzd17@tsinghua.org.cn)
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

import re
from typing import List
from tal_frontend.frontend.normalizer.cn.processor import Processor
#from processor import Processor
from tal_frontend.frontend.normalizer.cn.rules.cardinal import Cardinal
from tal_frontend.frontend.normalizer.cn.rules.char import Char
from tal_frontend.frontend.normalizer.cn.rules.date import Date
from tal_frontend.frontend.normalizer.cn.rules.fraction import Fraction
from tal_frontend.frontend.normalizer.cn.rules.math import Math
from tal_frontend.frontend.normalizer.cn.rules.measure import Measure
from tal_frontend.frontend.normalizer.cn.rules.money import Money
from tal_frontend.frontend.normalizer.cn.rules.postprocessor import PostProcessor
from tal_frontend.frontend.normalizer.cn.rules.preprocessor import PreProcessor
from tal_frontend.frontend.normalizer.cn.rules.sport import Sport
from tal_frontend.frontend.normalizer.cn.rules.time import Time
from tal_frontend.frontend.normalizer.cn.rules.whitelist import Whitelist


from pynini.lib.pynutil import add_weight, delete
from importlib_resources import files
from pathlib import Path


class cn_Normalizer(Processor):

    def __init__(self,
                 cache_dir=None,
                 overwrite_cache=False,
                 remove_interjections=True,
                 remove_erhua=False,
                 traditional_to_simple=True,
                 remove_puncts=False,
                 full_to_half=True,
                 tag_oov=False):
        super().__init__(name='normalizer')
        self.remove_interjections = remove_interjections
        self.remove_erhua = remove_erhua
        self.traditional_to_simple = traditional_to_simple
        self.remove_puncts = remove_puncts
        self.full_to_half = full_to_half
        self.tag_oov = tag_oov
        if cache_dir is None:
            script_path = Path(__file__).resolve()
            script_dir = script_path.parent
            cache_dir = script_dir
        self.build_fst('zh_tn', cache_dir, overwrite_cache)
        self.SENTENCE_SPLITOR = re.compile(r'([：、，；。？！,;?!][”’]?)')

    def build_tagger(self):
        processor = PreProcessor(
            traditional_to_simple=self.traditional_to_simple).processor

        date = add_weight(Date().tagger, 1.02)
        whitelist = add_weight(Whitelist().tagger, 1.03)
        sport = add_weight(Sport().tagger, 1.04)
        fraction = add_weight(Fraction().tagger, 1.05)
        measure = add_weight(Measure().tagger, 1.05)
        money = add_weight(Money().tagger, 1.05)
        time = add_weight(Time().tagger, 1.05)
        cardinal = add_weight(Cardinal().tagger, 1.06)
        math = add_weight(Math().tagger, 90)
        char = add_weight(Char().tagger, 100)
        # math = add_weight(Math().tagger, 1.05)
        # char = add_weight(Char().tagger, 1.05)

        tagger = (date | whitelist | sport | fraction | measure | money | time
                  | cardinal | math | char).optimize()
        tagger = (processor @ tagger).star
        # delete the last space
        self.tagger = tagger @ self.build_rule(delete(' '), r='[EOS]')

    def build_verbalizer(self):
        cardinal = Cardinal().verbalizer
        char = Char().verbalizer
        date = Date().verbalizer
        fraction = Fraction().verbalizer
        math = Math().verbalizer
        measure = Measure().verbalizer
        money = Money().verbalizer
        sport = Sport().verbalizer
        time = Time().verbalizer
        whitelist = Whitelist(remove_erhua=self.remove_erhua).verbalizer

        verbalizer = (cardinal | char | date | fraction | math | measure
                      | money | sport | time | whitelist).optimize()

        processor = PostProcessor(
            remove_interjections=self.remove_interjections,
            remove_puncts=self.remove_puncts,
            full_to_half=self.full_to_half,
            tag_oov=self.tag_oov).processor
        self.verbalizer = (verbalizer @ processor).star
    
    def _split(self, text: str, lang="zh") -> List[str]:
        if lang == "zh":
            text = text.replace(" ", "")
            # 过滤掉特殊字符
            text = re.sub(r'[——《》【】<=>{}()（）#&@“”^_|…\\]', '', text)
        text = self.SENTENCE_SPLITOR.sub(r'\1\n', text)
        text = text.strip()
        sentences = [sentence.strip() for sentence in re.split(r'\n+', text)]
        return sentences
