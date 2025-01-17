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

#from rules.cardinal import Cardinal
#from processor import Processor
from tal_frontend.frontend.normalizer.cn.rules.cardinal import Cardinal
from tal_frontend.frontend.normalizer.cn.processor import Processor
from pynini import accep, cross, string_file
from pynini.lib.pynutil import delete, insert, add_weight


class Measure(Processor):

    def __init__(self):
        super().__init__(name='measure')
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        units_en = string_file('tal_frontend/frontend/normalizer/cn/data/measure/units_en.tsv')
        units_zh = string_file('tal_frontend/frontend/normalizer/cn/data/measure/units_zh.tsv')
        units = add_weight((cross("k", "千") | cross("w", "万")), 0.1).ques + \
            (units_en | units_zh)
        rmspace = delete(' ').ques
        to = cross('-', '到') | cross('~', '到') |cross('～', '到')| cross('—', '到') | cross('－', '到') | accep('到')

        number = Cardinal().number
        number @= self.build_rule(cross('二', '两'), '[BOS]', '[EOS]')
        # 1-11个，1个-11个
        prefix = number + (rmspace + units).ques + to
        measure = prefix.ques + number + rmspace + units

        for unit in ['两', '月', '号']:
            measure @= self.build_rule(cross('两' + unit, '二' + unit),
                                       l='[BOS]')
            measure @= self.build_rule(cross('到两' + unit, '到二' + unit),
                                       r='[EOS]')

        # -xxxx年, -xx年
        digits = Cardinal().digits
        cardinal = digits**4
        # cardinal = digits**2 | digits**4
        unit = accep('年') |accep('年度') | accep('赛季') 
        prefix = cardinal + rmspace.ques + unit.ques + to
        annual = prefix.ques + cardinal + unit
        tagger = insert('value: "') + (measure | annual) + insert('"')

        # cardinal = digits**4
        # unit = accep('年')
        # prefix = rmspace.ques + unit + to
        # annual = prefix + cardinal + unit
        # tagger |= insert('value: "') + (measure | annual) + insert('"')
        
        # 公元前507年—前400年 公元前507年—公元前400年 公元前407年—公元前310年  2001年12月～2022年1月
        # cardinal = digits**4
        # early = accep('前') #| accep('公元前')
        # cardinal = digits.plus
        number = Cardinal().tri_numer
        early = accep('公元前') | accep('前') | accep('公元')
        unit = accep('年') | accep('日') | accep('月')
        prefix = early.ques + number + rmspace.ques + unit + to
        annual = prefix + early.ques + number + unit
        tagger |= insert('value: "') + (measure | annual) + insert('"')
        
        # 10km/h
        rmsign = rmspace + delete('/') + rmspace
        tagger |= (insert('numerator: "') + measure + rmsign +
                   insert('" denominator: "') + units + insert('"'))
        self.tagger = self.add_tokens(tagger)

    def build_verbalizer(self):
        super().build_verbalizer()
        denominator = delete('denominator: "') + self.SIGMA + delete('" ')
        numerator = delete('numerator: "') + self.SIGMA + delete('"')
        verbalizer = insert('每') + denominator + numerator
        # verbalizer = numerator + denominator
        self.verbalizer |= self.delete_tokens(verbalizer)
