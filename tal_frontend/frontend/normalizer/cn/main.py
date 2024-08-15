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

import argparse

# TODO(pzd17): multi-language support
#from cn.cn_normalizer import cn_Normalizer

#from cn.chinese.normalizer import Normalizer
from tal_frontend.frontend.normalizer.cn.cn_normalizer import cn_Normalizer
def str2bool(s, default=False):
    s = s.lower()
    if s == 'true':
        return True
    elif s == 'false':
        return False
    else:
        return default

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='input string')
    parser.add_argument('--file', help='input file path')
    parser.add_argument('--cache_dir',
                        type=str,
                        default=None,
                        help='cache dir containing *.fst')
    parser.add_argument('--overwrite_cache',
                        default=True,
                        action='store_true',
                        help='rebuild *.fst')
    parser.add_argument('--remove_interjections',
                        type=str,
                        default='False',
                        help='remove interjections like "啊"')
    parser.add_argument('--remove_erhua',
                        type=str,
                        default='False',
                        help='remove "儿"')
    parser.add_argument('--traditional_to_simple',
                        type=str,
                        default='False',
                        help='i.e., "喆" -> "哲"')
    parser.add_argument('--remove_puncts',
                        type=str,
                        default='False',
                        help='remove punctuations like "。" and "，"')
    parser.add_argument('--full_to_half',
                        type=str,
                        default='False',
                        help='i.e., "Ａ" -> "A"')
    parser.add_argument('--tag_oov',
                        type=str,
                        default='False',
                        help='tag OOV with "OOV"')
    args = parser.parse_args()
    normalizer = cn_Normalizer(
        cache_dir=args.cache_dir,
        overwrite_cache=args.overwrite_cache,
        remove_interjections=str2bool(args.remove_interjections),
        remove_erhua=str2bool(args.remove_erhua),
        traditional_to_simple=str2bool(args.traditional_to_simple),
        remove_puncts=str2bool(args.remove_puncts),
        full_to_half=str2bool(args.full_to_half),
        tag_oov=str2bool(args.tag_oov))

    if args.text:
        print(normalizer.tag(args.text))
        print(normalizer.normalize(args.text))
    elif args.file:
        with open(args.file) as fin:
            for line in fin:
                print(normalizer.tag(line.strip()))
                print(normalizer.normalize(line.strip()))


if __name__ == '__main__':
    main()

