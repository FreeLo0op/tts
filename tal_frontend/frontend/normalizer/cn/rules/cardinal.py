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

#from processor import Processor
from tal_frontend.frontend.normalizer.cn.processor import Processor


from pynini import accep, cross, string_file
from pynini.lib.pynutil import add_weight, delete, insert


class Cardinal(Processor):

    def __init__(self):
        super().__init__('cardinal')
        self.number = None
        self.digits = None
        self.tri_numer = None
        self.build_tagger()
        self.build_verbalizer()

    def build_tagger(self):
        zero = string_file('tal_frontend/frontend/normalizer/cn/data/number/zero.tsv')
        digit = string_file('tal_frontend/frontend/normalizer/cn/data/number/digit.tsv')
        teen = string_file('tal_frontend/frontend/normalizer/cn/data/number/teen.tsv')
        sign = string_file('tal_frontend/frontend/normalizer/cn/data/number/sign.tsv')
        dot = string_file('tal_frontend/frontend/normalizer/cn/data/number/dot.tsv')

        rmzero = delete('0') | delete('０')
        rmpunct = delete(',').ques
        digits = zero | digit
        self.digits = digits

        # 11 => 十一
        ten = teen + insert('十') + (digit | rmzero)
        # 11 => 一十一
        tens = digit + insert('十') + (digit | rmzero)
        # 111, 101, 100
        hundred = (digit + insert('百') + (tens | (zero + digit) | rmzero**2))
        # 1111, 1011, 1001, 1000
        thousand = (digit + insert('千') + rmpunct + (hundred
                                                     | (zero + tens)
                                                     | (rmzero + zero + digit)
                                                     | rmzero**3))
        # 10001111, 1001111, 101111, 11111, 10111, 10011, 10001, 10000
        ten_thousand = ((thousand  | hundred  | ten | digit)  + insert('万') + rmpunct + 
                        (thousand
                         | (zero + rmpunct + hundred)
                         | (rmzero + rmpunct + zero + tens)
                         | (rmzero + rmpunct + rmzero + zero + digit)
                         | rmzero**4))        

        # 11,1111  111,1111
        # hundred_million = (
        #     (digit) + insert('亿') + rmpunct + (
        #         thousand + insert('万') + rmpunct + thousand + rmpunct 
        #         | (rmzero + ten_thousand)
        #         | (rmzero**2 + ten_thousand)
        #         | (rmzero**3 + ten_thousand)
        #         | (rmzero**4 + rmpunct  + thousand)
        #         | (rmzero**4 + rmpunct + rmzero + hundred)
        #         | (rmzero**4 + rmpunct + rmzero**2 + tens)
        #         | (rmzero**4 + rmpunct + rmzero**3 + digit)
        #         | (rmzero**4 + rmpunct + rmzero**4 + rmpunct)
        #         | rmzero**8
        #     )
        # )

        hundred_million = (
            (digit) + insert('亿') + rmpunct + (
                ((thousand  | hundred  | ten | digit)  + insert('万'))  + rmpunct 
                + (thousand|hundred|ten|digit)  )
        )
        # 1.11, 1.01
        number = digits | ten | hundred | thousand | ten_thousand# | hundred_million
        t_number = digits| ten| hundred | digits**4
        self.tri_numer = t_number
        number = sign.ques + number + (dot + digits.plus).ques
        number @= self.build_rule(
            cross('二百', '两百')
            | cross('二千', '两千')
            | cross('十二万', '十二万')
            | cross('二万', '两万')).optimize()
        percent = insert('百分之') + number + delete('%')
        ten_percent = insert('千分之') + number + delete('‰')
        
        self.number = accep('约').ques + accep('人均').ques + (number | percent | ten_percent)

        # cardinal string like 127.0.0.1, used in ID, IP, etc.
        cardinal = digits.plus + (dot + digits.plus)**3
        cardinal |= percent
        cardinal |= ten_percent
        # xxxx-xxx-xxx
        cardinal |= digits.plus + (delete('-') + digits.plus)**2
        # xxx-xxxxxxxx
        cardinal |= digits**3 + delete('-') + digits**8
        # three or five or eleven phone numbers
        phone_digits = digits @ self.build_rule(cross('一', '幺'))
        # phone = phone_digits**3 | phone_digits**5 | phone_digits**11
        phone = phone_digits**11
        phone |= accep("尾号") + (accep("是") | accep("为")).ques + phone_digits**4
        cardinal |= add_weight(phone, -1.0)

        tagger = insert('value: "') + cardinal + insert('"')
        self.tagger = self.add_tokens(tagger)
