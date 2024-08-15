import re
import jieba
from collections import defaultdict

from tal_frontend.frontend.g2p.phonemes.en.en_phoneme_len import en_phoneme_len_dict
# from tal_frontend.frontend.normalizer.cn.cn_normalizer import cn_Normalizer as cn_Normalizer
from tal_frontend.frontend.normalizer.en.en_normalizer import en_normalize
from tal_frontend.frontend.normalizer.symblos import *
from tal_frontend.utils.special_words import special_words

class TextProcessor:
    def __init__(self, cn_normalizer,senlen: int=30, en_phoneme_ave_len: int=6) -> None:
        self.illegal_patterns = illegal_patterns
        self.zh_normalizer = cn_normalizer
        self.en_normalizer = en_normalize
        self.senlen = senlen
        self.en_phoneme_ave_len = en_phoneme_ave_len
    
    def rmove_continus_punks(self, text:str):
        clean_text = re.sub(r'([\!\@\#\$\%\^\&\*\(\)\_\[\{\]\}\\\|\;\:\'\,\.\<\>\?\！\@\#\¥\%\…\&\*\（\）\「\【\『\「\」\』\】\、\｜\；\：\‘\’\“\”\，\。\《\》\？\—])[\!\@\#\$\%\^\&\*\(\)\_\[\{\]\}\\\|\;\:\'\,\.\<\>\?\！\@\#\¥\%\…\&\*\（\）\「\【\『\「\」\』\】\、\｜\；\：\‘\’\“\”\，\。\《\》\？\—]+', r'\1', text)
        return clean_text
    
    def text_clean(self, text: str) -> str:
        
        #half_text = ''
        #for i in clean_text:
        #    half_text += self.full2half_width(i)
        #return half_text
        #clean_text = self.remove_spaces_between_cn_en(clean_text)
        text = self.special_pron(text)
        # 移除非法字符
        clean_text = re.sub(self.illegal_patterns, '', text)
        # 数据库中的脏数据 <b>
        clean_text = re.sub(r'\<b\>', '', clean_text)
        # clean_text = re.sub(r'([\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/\:\;\<\=\>\?\@\[\]\\\^\_\`\{\|\}\~～！，。？《》/：；”“’‘【】「」！¥……（）——])[\!\"\#\$\%\&\'\(\)\*\+\,\-\.\/:;<=>\?\@\[\]\\\^_`{|}~～！，。？《》/：；”“’‘【】「」！¥……（）——]+', r'\1', clean_text)
        return clean_text
    
    def special_pron(self, text:str):
        ### 从数据库获取文本的发音
        for key, value in special_words.items():
            text = re.sub(key, value, text)
        return text

    def en_syllable_extraction(self, lines):
        en_dict = defaultdict(set)
        for i in range(0, len(lines), 2):
            key, content = lines[i].strip().split('\t', maxsplit=1)
            syllables = lines[i+1].strip()
            words = re.findall(r'[a-zA-z\'\.\-]+', content)
            en_syllables = re.findall(r'[A-Z]+[A-Z \.0-4]+', syllables)
            if len(words) != len(en_syllables):
                print(f'{key}\t{content}\t{syllables}')
            else:
                for word, en_syllable in zip(words, en_syllables):
                    en_dict[word].add(en_syllable)
        return en_dict
    
    '''
    def text_detection(self, text: str) -> list:
        # 检测正常字符串和latex数学公式
        parts = re.split(math_match_pattern, text)
        result = []
        for part in parts:
            # True 表示 LaTeX 部分
            # False 表示非 LaTeX 部分
            try:
                if re.match(math_match_pattern, part):
                    clean_latex = re.sub(r'\$','',part)
                    half_text = ''
                    for i in clean_latex:
                        half_text += self.full2half_width(i)
                    latex_read = self.formula2text('latex', half_text)
                    result.append(latex_read)
                else:
                    result.append(part)
            except Exception as e:
                raise ValueError(f'Latex公式解析失败 {e}')
        result = ' '.join(result)
        #result = self.remove_spaces_between_cn_en(result) 
        return result
    '''
    
    def full2half_width(slef, ustr):
        # 全角转半角
        num = ord(ustr)
        if num == 0x3000:  # 全角空格变半角
            num = 32
        #if re.match(en_punct_pattern, ustr) or re.match(cn_punct_pattern, ustr):
        #    return ustr
        if 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        return chr(num)
    
    def split_by_lang(self, text: str, lang: str = 'cn') -> list[tuple]:
        if lang not in ['cn', 'en']:
            raise ValueError('非法语言类型. 请在 "cn" 或 "en" 中选择语种.')
        segments = []
        for ch in text:
            if lang == 'cn':
                # if re.match(cn_pattern, ch) or re.match(cn_punct_pattern, ch) or re.match(num_pattern, ch):
                if re.match(cn_pattern, ch) or re.match(cn_punct_pattern, ch):
                    segments.append((ch, 'cn'))
                # elif re.match(r'[a-zA-Z]', ch):
                # # elif re.match(r'[a-zA-Z\,\?\:\!\(\)]', ch):
                #     segments.append((ch, 'en'))
                else:
                    segments.append((ch, 'other'))
            elif lang == 'en':
                if re.match(cn_pattern, ch):
                    raise ValueError(f"非法字符：预期为英文，但找到中文字符 '{ch}' ")
                else:
                    segments.append((ch, 'en'))
        
        # 如果第一个字符是 'other'，将其设置为 'cn'
        if segments and segments[0][1] == 'other':
            segments[0] = (segments[0][0], 'cn')
        
        # 合并字符基于他们的类型
        merged_segments = []
        temp_seg = segments[0][0]
        temp_lang = segments[0][1]
        
        for i in range(1, len(segments)):
            current_char, current_lang = segments[i]
            if current_lang == 'other':
                # 检查前后文语种
                prev_lang = temp_lang
                current_lang = prev_lang
                # next_lang = segments[i+1][1] if i+1 < len(segments) else 'cn'
                # if prev_lang == 'en' and i == len(segments)-1:
                #     current_lang = 'en'
                # elif prev_lang == 'en' and next_lang == 'en':
                #     current_lang = 'en'
                # elif prev_lang == 'en' and next_lang == 'other':
                #     current_lang = 'en'
                # else:
                #     current_lang = 'cn'
            
            if current_lang == temp_lang:
                temp_seg += current_char
            else:
                merged_segments.append((temp_seg, temp_lang))
                temp_seg = current_char
                temp_lang = current_lang
        
        # Append the last segment
        merged_segments.append((temp_seg, temp_lang))
        
        return merged_segments
        
    def text_normalization(self, text, lang):
        if lang == 'cn':
            try:
                nor_text = text.replace("i.e.", "that is")
                nor_text = nor_text.replace("e.g.", "for example")
                nor_text = re.sub(r'([a-zA-Z]+)[\']+([a-zA-Z]+)',r'\1\2', nor_text)
                nor_text = re.sub(r'([a-zA-Z]+)[\-]+([a-zA-Z]+)',r'\1 \2', nor_text)
                nor_text = self.zh_normalizer.normalize(nor_text)
                #nor_text = re.sub(cn_illegal_patterns,'',nor_text)
                # nor_text = self.rmove_continus_punks(nor_text)
            except Exception as e:
                raise ValueError(f'中文文本标准化失败，错误原因{e}')
        elif lang == 'en':
            try:
                nor_text = self.en_normalizer(text)
                nor_text = re.sub(en_illegal_patterns, ' ', nor_text)
            except Exception as e:
                raise ValueError(f'英文文本标准化失败，错误原因{e}')
        else:
            raise ValueError(f'非法语种 {lang}')
        return nor_text
    
    # 移除多余的空格
    def remove_spaces_between_cn_en(self, text):
        punctuations = r'\.\,\?\:\!\，\。\；\：\、\！\？\"'
        text = re.sub(r'[ ]+', ' ', text)
        
        # 移除标点符号前后的空格
        text = re.sub(r'\s+([{}])'.format(re.escape(punctuations)), r'\1', text)
        #text = re.sub(r'([{}])\s+'.format(re.escape(punctuations)), r'\1', text)
        
        # 移除中文和英文字符之间的空格
        text = re.sub(r'([\u4e00-\u9fff]) (\w)', r'\1\2', text)
        text = re.sub(r'(\w) ([\u4e00-\u9fff])', r'\1\2', text)
        text = re.sub(r'[ ]?#9[ ]?', '#9', text)
        return text.strip(' ')
    
    def en_rhy(self, text):
        # 添加英文韵律
        # 空格处添加 #1
        # 中英文交接处或没有标点符号的句尾添加 #2
        # 标点符号后面跟 #3
        #text = re.sub(en_punct_pattern, r'\g<0>#3', text)
        # 处理特定缩写，如 "mr.", "mrs.", "dr."
        #text = re.sub(r'\b(mr|mrs|dr)\.\#3', r'\1.', text, flags=re.IGNORECASE)
        #text = re.sub(r'(?<!#3) ', '#1', text)
        #if not text.endswith('#3'):
        #    text += '#2'
        # 处理特定缩写，确保缩写不被错误处理
        text = re.sub(r'\b(mr|mrs|dr)\.', r'\1.', text, flags=re.IGNORECASE)

        # 标记 #9 后面的空格，使其不被错误处理
        text = re.sub(r'(#9\s+)', r'#9#9temp', text)

        # 添加一个特殊标记到英文单词之间的空格
        #text = re.sub(r'([a-z]+[\'\-]?[a-z]*|[a-z]) ([a-z]+[\'\-]?[a-z]*|[a-z])', r'\1#1\2', text, flags=re.IGNORECASE)
        # 替换特殊标记为 #1
        #text = re.sub(r'##SPECIAL##', '#1 ', text)
        

        # 恢复 #9 后的空格
        text = re.sub(r'#9#9temp', r'#9 ', text)

        # 中英文交接处添加 #2
        text = re.sub(r'([a-zA-Z]) ([\u4e00-\u9fff])', r'\1#2 \2', text)
        text = re.sub(r'([\u4e00-\u9fff]) ([a-zA-Z])', r'\1#2 \2', text)

        # 英文标点符号前面如果是英文，添加 #3
        text = re.sub(r'([a-zA-Z])([\.\,\?\:\!])', r'\1#3\2', text)

        text = re.sub(r'([a-z]) ', r'\1#1 ', text)

        return text
    
    def split_sentence_by_phoneme(self, text:str, phonemes:str):
        phoneme_list = phonemes.strip('/').split('/')
        text = re.sub(all_punct_pattern, '', text)
        text_prosody_list = re.findall(r'[\u4e00-\u9fff]|[a-z]+[\']?[a-z]+|[a-z]|#1|#2|#3|#9', text) 
        
        # 确保文本长度和音素长度一致
        text_list = re.findall(r'[\u4e00-\u9fff]|[a-z]+[\']?[a-z]+|[a-z]', text)
        #print(phonemes)
        #print(text_list)
        #print(phoneme_list)
        assert len(text_list) == len(phoneme_list)
        # 插入韵律信息到音素列表中
        result_phonemes = []
        text_index = 0
        
        for item in text_prosody_list:
            if re.match(r'#1|#2|#3|#9', item):
                result_phonemes.append(item)
            else:
                result_phonemes.append(phoneme_list[text_index])
                text_index += 1

        result_phonemes = ' '.join(result_phonemes)
        result_phonemes = re.sub(r'[ ]+', ' ', result_phonemes)
        # 递归地用二分法切分长于20的片段
        def split_by_delimiter(segment, delimiter):
            positions = [m.start() for m in re.finditer(delimiter, segment)]
            if not positions:
                # 没有找到分隔符时直接二分
                mid = len(segment) // 2
                return split_by_sp(segment[:mid]) + split_by_sp(segment[mid:])
            else:
                # 找到最接近中间的分隔符位置
                mid = len(segment) // 2
                closest_pos = min(positions, key=lambda x: abs(x - mid))
                left_segment = segment[:closest_pos]
                right_segment = segment[closest_pos + len(delimiter):]
                #print('左',left_segment)
                #print('右',right_segment)
                return split_by_sp(left_segment) + split_by_sp(right_segment)

        def split_by_sp(segment: str):
            segment = segment.strip()
            length = len(re.sub(r'#1', '', segment).split())
            if length <= 20:
                return [re.sub(r'#1|#2|#3|#9', '', segment)]
            elif '#2' in segment:
                return split_by_delimiter(segment, '#2')
            elif '#1' in segment:
                return split_by_delimiter(segment, '#1')
            else:
                # 没有 #1 和 #2 分隔符时直接二分
                mid = len(segment) // 2
                return split_by_sp(segment[:mid]) + split_by_sp(segment[mid:])

        split_phonemes = []
        split_sp3s = ''.join(result_phonemes).split('#3')
        for split_sp3 in split_sp3s:
            if split_sp3:
                split_segments = split_by_sp(split_sp3)
                split_phonemes.extend(split_segments)
        # 合并后的逻辑：将生成的结果与原始音素列表进行对齐
        single_ph = result_phonemes.split()

        res = []
        start = 0

        for phoneme_segment in split_phonemes:
            if phoneme_segment:
                phoneme_segment = phoneme_segment.strip().split()
                index_1, index_2 = start, 0
                tmp_res = []
          
                while index_2 < len(phoneme_segment):
                    if index_1 < len(single_ph):
                        if phoneme_segment[index_2] == single_ph[index_1]:
                            tmp_res.append(phoneme_segment[index_2])
                            index_2 += 1
                        else:
                            tmp_res.append(single_ph[index_1])
                        index_1 += 1

                if index_1 < len(single_ph) and single_ph[index_1] in ['#1', '#2', '#3', '#9']:
                    tmp_res.append(single_ph[index_1])
                    index_1 += 1
                
                res.append(' '.join(tmp_res))
                start = index_1
        
        '''
        res = []
        start = 0
        for i in split_phonemes:
            i = i.strip().split()
            index_1, index_2 = start, 0
            tmp_res = []
            while index_2 < len(i):
                if i[index_2] == single_ph[index_1]:
                    tmp_res.append(i[index_2])
                    index_1 += 1
                    index_2 += 1
                else:
                    tmp_res.append(single_ph[index_1])
                    index_1 += 1
            if index_1 < len(single_ph) and single_ph[index_1] in ['#1', '#2', '#3', '#9']:
                tmp_res.append(single_ph[index_1])
                index_1 += 1
            res.append(' '.join(tmp_res))
            start = index_1]
        '''
        return res
    
    def sentence_split_cn(self, text:str, )-> list[list]:
        split_words = jieba.lcut(text)
        split_sentencs = []
        res, length = '', 0
        for word in split_words:
            if re.search(r'[ ：、]', word):
                res += word
            elif re.search(r'[。！？，；,\.\!\?]', word):
                res += word
                split_sentencs.append([res, length*2])
                length = 0
                res = ''
            else:
                word_len = len(word)
                tmp_len = length + word_len
                if tmp_len <= self.senlen:
                    length = tmp_len
                    res += word
                else:
                    split_sentencs.append([res, length*2])
                    res = word
                    length = word_len
        if res:
            split_sentencs.append([res, length*2])
        return split_sentencs

    def sentence_split_en(self, text:str)-> list[list]:
        # split_words = re.findall(r'[a-z]+[\'\-]?[a-z]+|[a-z]|[\.\,\?\:\!\;]', text)
        split_words = re.findall(r'[a-zA-Z]+|[a-zA-Z]|[\.\,\?\:\!\;]', text)
        res, length = '', 0
        split_sentencs = []
        for word in split_words:
            if re.match(r'[\,\:\;]', word):
                res += word
            elif re.match(r'[\.\?\!]', word):
                res += word
                split_sentencs.append([res, length])
                length = 0
                res = ''
            else:
                try:
                    word_len = en_phoneme_len_dict[word]
                except:
                    word_len = self.en_phoneme_ave_len # 取cmu英文字典平均音素长度，能覆盖57.6%的英文
                tmp_len = length + word_len
                if tmp_len <= self.senlen*2:
                    length = tmp_len
                    res += f'{word} '
                else:
                    split_sentencs.append([res.strip(), length])
                    res = f'{word} '
                    length = word_len
        if res:
            split_sentencs.append([res.strip(), length])
        return split_sentencs
    
    def sentence_split_mix(self, text:str, )-> list[list]:
        split_words = jieba.lcut(text)
        split_sentencs = []
        res, length = '', 0
        for word in split_words:
            if re.search(r'[ ：、\,\:\;]', word):
                res += word
            elif re.search(r'[。！？，；,\.\!\?]', word):
                res += word
                split_sentencs.append([res, length*2])
                length = 0
                res = ''
            elif re.search(r'[a-z]+', word):
                try:
                    word_len = en_phoneme_len_dict[word]
                except:
                    word_len = self.en_phoneme_ave_len # 取cmu英文字典平均音素长度，能覆盖57.6%的英文
                tmp_len = length + word_len
                if tmp_len <= self.senlen*2:
                    length = tmp_len
                    res += f'{word}'
                else:
                    split_sentencs.append([res.strip(), length])
                    res = f'{word}'
                    length = word_len
            else:
                word_len = len(word)*2
                tmp_len = length + word_len
                if tmp_len <= self.senlen * 2:
                    length = tmp_len
                    res += word
                else:
                    split_sentencs.append([res, length])
                    res = word
                    length = word_len
        if res:
            split_sentencs.append([res, length])
        return split_sentencs
    def merge_short_sentences(self, sentences:list[list]) -> list[list]:
        merged_sentencs = [sentences[0]]
        for i in range(1, len(sentences)):
            sentence, length = sentences[i]
            sentence = self.remove_spaces_between_cn_en(sentence)
            if sentence:
                last_sen_len = merged_sentencs[-1][1]
                if (last_sen_len + length) <= self.senlen * 2:
                    merged_sentencs[-1][0] += f' {sentence}'
                    merged_sentencs[-1][1] += length
                else:
                    merged_sentencs.append(sentences[i])
        return merged_sentencs