import re
import time

from tal_frontend.frontend.g2p.utils import ph2id,rhy2id
from tal_frontend.frontend.normalizer.textprocesser import TextProcessor
from tal_frontend.frontend.ssml.xml2text import xml_reader, xml_reader_string
from clients.utils import math_request
from .utils.vits_config import *
from .utils.errors import *
from tools.error_config import ErrorCode


class Frontend:
    def __init__(self, 
                 cn_normalizer,
                 formular2text_url:str='http://123.57.25.89:8023/inference', 
                 senlen:int=30, 
                 en_phoneme_ave_len:int=6, 
                 ):
        self.speak_info = DEFAULT_SPEAK_INFO.copy()
        self.speak_id = SPK_ID.copy()
        self.vits_model = VITS_MODEL.copy()
        
        
        self.tp = TextProcessor(cn_normalizer,senlen=senlen, en_phoneme_ave_len=en_phoneme_ave_len)
        self.formular2text_url = formular2text_url
      

    def params_check(self, params: dict):
        def validate_param(value, valid_values=None, value_range=None):
            if valid_values and value not in valid_values:
                return False, value
            if value_range:
                try:
                    if not (value_range[0] <= float(value) <= value_range[1]):
                        return False, value
                except ValueError:
                    return False, value
            return True, ''

        for param, value in params.items():
            valid_values = VALID_PARAMS[param]['values']
            value_range = VALID_PARAMS[param]['range']
            is_valid, error_message = validate_param(
                value, valid_values, value_range)
            #if param == 'volume':
            #    self.speak_info[param] = float(self.speak_info[param]) / 100
            #elif param == 'rate':
            #    self.speak_info[param] = 1/ float(self.speak_info[param])
            if not is_valid:
                raise InvalidParameterError(
                    f'{param} 参数错误，错误值：{error_message}')

        spk = f"{params['voice_type']}_{params['lang']}_{params['emotion']}"
        if spk not in self.speak_id:
            raise InvalidParameterError(
                    f'参数错误，错误值：{spk}')
        
    def process_text(self, text: str, g2p_pp, voice_params: dict,  math_type,):
        
        process_text_memory = {"process_route":"process_text"}
        
        for key, value in voice_params.items():
            if value:
                self.speak_info[key] = value
        # logger.info(code=ErrorCode.SUCCESS.value, message={"tag":"process_text开始", "text": text, "voice_params": voice_params}, trace_id=trace_id)
 
                
        self.params_check(self.speak_info)
        cleaned_text = self.tp.text_clean(text)
        
        process_text_memory['cleaned_text'] = cleaned_text
        
        if math_type:
            detected_text = math_request(
                cleaned_text, self.formular2text_url, math_type)
            process_text_memory['latex2text'] = detected_text
            detected_text = self.tp.remove_spaces_between_cn_en(detected_text)

        else:
            detected_text = cleaned_text
            

        segments = self.tp.split_by_lang(
            detected_text, lang=self.speak_info['lang'])
        # print(segments)
        long_sentences = []
        for seg in segments:
            text, lang = seg
            # if not re.search(r'[a-zA-Z\u4e00-\u9fff0-9]', text):
            #     long_sentences.append([text, 0])
            #     continue
            nor_text = self.tp.text_normalization(text, lang)
            if lang == 'cn':
                # split_sentences = self.tp.sentence_split_cn(nor_text)
                split_sentences = self.tp.sentence_split_mix(nor_text)
            elif lang == 'en':
                split_sentences = self.tp.sentence_split_en(nor_text)
            long_sentences.extend(split_sentences)
        long_sentences = self.tp.merge_short_sentences(long_sentences)
        # print(f"==>> long_sentences: {long_sentences}")
        process_text_memory['long_sentences'] = long_sentences

        info4vits, phonemes, prosodies = [], [], []
        nor_texts = [item[0] for item in long_sentences]
        
        # return '/'.join(nor_texts), '', '', '', ''
        prosodys_, phonemes_ = g2p_pp.infer(nor_texts)

        for i in range(len(nor_texts)):
            phonemes_[i] = ['sil'] + phonemes_[i]
            phonemes_id = ph2id(phonemes_[i])
            prosodies_id = rhy2id(prosodys_[i])
            if len(prosodies_id) == len(phonemes_id) + 1:
                prosodies_id = prosodies_id[:-1]
                prosodies_id[-1] = 1
                prosodies = prosodies[:-1]
                prosodies_id[-2] = 6
            phonemes.extend(phonemes_[i])
            prosodies.extend(prosodys_[i])
            if len(phonemes_id) > 70:
                self.speak_info['Warning'] = '合成文本过长，仅保留前70个音素。'
                
            spk_id_key = f"{self.speak_info['voice_type']}_{self.speak_info['lang']}_{self.speak_info['emotion']}"
            spk_id = self.speak_id[spk_id_key]
            tts_type = self.vits_model[spk_id_key]
            
            info = [spk_id, tts_type, i+1, phonemes_id, self.speak_info['rate'],
                    self.speak_info['pitch'], self.speak_info['volume'], {}, prosodies_id]
            info4vits.append(info)
            # logger.info(code=ErrorCode.SUCCESS.value, message={"tag":"process_text结束", "text": text, "voice_params": voice_params}, trace_id=trace_id)

        return '/'.join(nor_texts), ' '.join(phonemes), ' '.join(prosodies), info4vits, self.speak_info,process_text_memory

    def process_ssml(self, xml, g2p_pp, type='string'):
        if type == 'string':
            contents, speak_info = xml_reader_string(xml)
        else:
            contents, speak_info = xml_reader(xml)
        for key, value in speak_info.items():
            if value:
                self.speak_info[key] = value
        self.params_check(self.speak_info)
        sentences, break_lists = [], []
        for content in contents:
            c_type, c_addition, c_text = content
            match c_type:
                case 'pinyin':
                    sentence = ' '.join(c_text)
                    phonemes = c_addition.strip('/').split('/')
                    sentences.append([sentence.lower(), phonemes])
                    continue
                case 'break':
                    sentences.append(['#9', 'silence'])
                    break_lists.append(c_text)
                    continue
                case 'math':
                    c_text = math_request(f'${c_text}$',self.formular2text_url, c_addition)

                    c_text = self.tp.remove_spaces_between_cn_en(c_text)

                case 'text':
                    pass
            cleaned_text = self.tp.text_clean(c_text)

            segments = self.tp.split_by_lang(cleaned_text, lang='cn')
            for seg in segments:
                text, lang = seg
                # if not re.search(r'[a-zA-Z\u4e00-\u9fff0-9]', text):
                # if re.search(r'[!@#$%^&*()_+-={}[]\\]')
                #     sentences.append([text, 0])
                #     continue
                nor_text = self.tp.text_normalization(text, lang)
                if lang == 'cn':
                    split_sentences = self.tp.sentence_split_cn(nor_text)
                elif lang == 'en':
                    split_sentences = self.tp.sentence_split_en(nor_text)
                sentences.extend(split_sentences)
  
        sentences.append(['#9', 'silence'])
        sentences_len, count = len(sentences), 0
        merged_sentences, tmp = [], []
        phonemes, prosodies, nor_texts = [], [], []
        while count < sentences_len:
            content, flag = sentences[count]
            nor_texts.append(content)
            if isinstance(flag, int):
                tmp.append(sentences[count])
            else:
                if not tmp:
                    if flag == 'silence':
                        merged_sentences.append(sentences[count])
                    else:
                        prosody, phoneme = g2p_pp.infer([content])
                        prosody, phoneme = prosody[0], phoneme[0]
                        phoneme = [item.strip()
                                   for sublist in flag for item in sublist.split()]
                        phonemes.extend(phoneme)
                        prosodies.extend(prosody)
                        phoneme = ['sil'] + phoneme
                        phonemes_id = ph2id(phoneme)
                        prosodies_id = rhy2id(prosody)
                        merged_sentences.append([phonemes_id, prosodies_id])
                else:
                    tmp = self.tp.merge_short_sentences(tmp)
                    for sentence in tmp:
                        prosody, phoneme = g2p_pp.infer([sentence[0]])
                        prosody, phoneme = prosody[0], phoneme[0]
                        phoneme = ['sil'] + phoneme
                        phonemes_id = ph2id(phoneme)
                        prosodies_id = rhy2id(prosody)
                        
                        merged_sentences.append([phonemes_id, prosodies_id])
                        phonemes.extend(phoneme)
                        prosodies.extend(prosody)
                    if flag == 'silence':
                        merged_sentences.append(sentences[count])
                    else:
                        prosody, phoneme = g2p_pp.infer([content])
                        prosody, phoneme = prosody[0], phoneme[0]
                        phoneme = [item.strip()
                                   for sublist in flag for item in sublist.split()]
                        phoneme = ['sil'] + phoneme
                        phonemes_id = ph2id(phoneme)
                        prosodies_id = rhy2id(prosody)
                        
                        merged_sentences.append([phonemes_id, prosodies_id])
                        phonemes.extend(phoneme)
                        prosodies.extend(prosody)
                    tmp = []
            count += 1
        merged_sentences = merged_sentences[:-1]
        info4vits, i = [], 0
        while i < len(merged_sentences)-1:
            content = merged_sentences[i]
            break_list = {}
            if content[1] == 'silence':
                content = merged_sentences[i+1]
                break_time = break_lists.pop(0)
                break_time = int(break_time.replace('ms', ''))//10
                break_list[0] = break_time
                i += 1
            if len(content[0]) > 70:
                self.speak_info['Warning'] = '合成文本过长，仅保留前70个音素。'
            spk_id_key = f"{self.speak_info['voice_type']}_{self.speak_info['lang']}_{self.speak_info['emotion']}"
            spk_id = self.speak_id[spk_id_key]
            tts_type = self.vits_model[spk_id_key]
            info = [spk_id, tts_type, len(info4vits)+1, content[0], self.speak_info['rate'],
                    self.speak_info['pitch'], self.speak_info['volume'], break_list, content[1]]
            info4vits.append(info)
            i += 1
        if i == len(merged_sentences)-1:
            content = merged_sentences[i]
            if content[1] == 'silence':
                phonemes_id = info4vits[-1][3]
                prosodies_id = info4vits[-1][-1]
                if phonemes_id[-2] == 7:
                    break_time = break_lists.pop(0)
                    break_time = int(break_time.replace('ms', ''))//10
                    break_list = {len(phonemes_id)-2: break_time}
                elif phonemes_id[-2] == 6:
                    phonemes_id[-2] = 7
                    prosodies_id[-2] = 5
                    break_time = break_lists.pop(0)
                    break_time = int(break_time.replace('ms', ''))//10
                    break_list[len(phonemes_id)-1] = break_time
                else:
                    break_time = break_lists.pop(0)
                    break_time = int(break_time.replace('ms', ''))//10
                    break_list[len(phonemes_id)-1] = break_time
                info4vits[-1][-2] = break_list
            else:
                if len(content[0]) > 70:
                    self.speak_info['Warning'] = '合成文本过长，仅保留前70个音素。'
                break_list = {}
                spk_id_key = f"{self.speak_info['voice_type']}_{self.speak_info['lang']}_{self.speak_info['emotion']}"
                spk_id = self.speak_id[spk_id_key]
                tts_type = self.vits_model[spk_id_key]
                info = [spk_id, tts_type, len(info4vits)+1, content[0], self.speak_info['rate'],
                        self.speak_info['pitch'], self.speak_info['volume'], {}, content[1]]
                info4vits.append(info)
        
        return ' '.join(nor_texts[:-1]), ' '.join(phonemes), ' '.join(prosodies), info4vits, self.speak_info
    
    def process_ssml2(self, xml, g2p_pp, type='string'):
        process_text_memory = {"process_route":"process_text"}

        if type == 'string':
            contents, speak_info = xml_reader_string(xml)
            process_text_memory['contents'] = contents
        else:
            contents, speak_info = xml_reader(xml)
            process_text_memory['contents'] = contents

        for key, value in speak_info.items():
            if value:
                self.speak_info[key] = value
        self.params_check(self.speak_info)
        sentences, break_lists = [], []
        for content in contents:
            c_type, c_addition, c_text = content
            match c_type:
                case 'pinyin':
                    sentence = ' '.join(c_text)
                    phonemes = c_addition.strip('/').split('/')
                    sentences.append([sentence.lower(), phonemes])
                    continue
                case 'break':
                    sentences.append(['#9', 'silence'])
                    break_lists.append(c_text)
                    continue
                case 'math':
                    
                    
                    
                    c_text = math_request(f'${c_text}$',self.formular2text_url, c_addition)
                    process_text_memory['latex2text'] = c_text

                    c_text = self.tp.remove_spaces_between_cn_en(c_text)
                case 'text':
                    pass
            cleaned_text = self.tp.text_clean(c_text)
            
            process_text_memory['cleaned_text'] = cleaned_text
            
            segments = self.tp.split_by_lang(cleaned_text, lang='cn')
            for seg in segments:
                text, lang = seg
                # if not re.search(r'[a-zA-Z\u4e00-\u9fff0-9]', text):
                # if re.search(r'[!@#$%^&*()_+-={}[]\\]')
                #     sentences.append([text, 0])
                #     continue
                nor_text = self.tp.text_normalization(text, lang)
                if lang == 'cn':
                    split_sentences = self.tp.sentence_split_cn(nor_text)
                elif lang == 'en':
                    split_sentences = self.tp.sentence_split_en(nor_text)
                sentences.extend(split_sentences)
  
        sentences.append(['#9', 'silence'])
        sentences_len, count = len(sentences), 0
        merged_sentences, tmp, define_pho = [], [], {}
        phonemes, prosodies, nor_texts = [], [], []
        while count < sentences_len:
            content, flag = sentences[count]
            nor_texts.append(content)
            if isinstance(flag, int):
                tmp.append(sentences[count])
            elif isinstance(flag, list):
                # ['顿 衣 服', ['d uen4', 'i1', 'f u2']]
                pho_len = sum([len(dp) for dp in sentences[count][1]])
                for tmp_word, tmp_pho in zip(sentences[count][0].split(), sentences[count][1]):
                    define_pho[tmp_word] = re.sub(r'\s', '', tmp_pho)
                sentences[count][1] = pho_len
                tmp.append(sentences[count])
            else:
                if not tmp:
                    if flag == 'silence':
                        merged_sentences.append(sentences[count])
                    else:
                        prosody, phoneme = g2p_pp.infer([content])
                        prosody, phoneme = prosody[0], phoneme[0]
                        phoneme = [item.strip()
                                   for sublist in flag for item in sublist.split()]
                        phonemes.extend(phoneme)
                        prosodies.extend(prosody)
                        phoneme = ['sil'] + phoneme
                        phonemes_id = ph2id(phoneme)
                        prosodies_id = rhy2id(prosody)
                        merged_sentences.append([phonemes_id, prosodies_id])
                else:
                    tmp = self.tp.merge_short_sentences(tmp)
                    for sentence in tmp:
                        prosody, phoneme = g2p_pp.infer([sentence[0]], define_pho=define_pho)
                        define_pho = {}
                        prosody, phoneme = prosody[0], phoneme[0]
                        phoneme = ['sil'] + phoneme
                        phonemes_id = ph2id(phoneme)
                        prosodies_id = rhy2id(prosody)
                        
                        merged_sentences.append([phonemes_id, prosodies_id])
                        phonemes.extend(phoneme)
                        prosodies.extend(prosody)
                    if flag == 'silence':
                        merged_sentences.append(sentences[count])
                    else:
                        prosody, phoneme = g2p_pp.infer([content])
                        prosody, phoneme = prosody[0], phoneme[0]
                        phoneme = [item.strip()
                                   for sublist in flag for item in sublist.split()]
                        phoneme = ['sil'] + phoneme
                        phonemes_id = ph2id(phoneme)
                        prosodies_id = rhy2id(prosody)
                        
                        merged_sentences.append([phonemes_id, prosodies_id])
                        phonemes.extend(phoneme)
                        prosodies.extend(prosody)
                    tmp = []
            count += 1
        merged_sentences = merged_sentences[:-1]
        info4vits, i = [], 0
        while i < len(merged_sentences)-1:
            content = merged_sentences[i]
            break_list = {}
            if content[1] == 'silence':
                content = merged_sentences[i+1]
                break_time = break_lists.pop(0)
                break_time = int(break_time.replace('ms', ''))//10
                break_list[0] = break_time
                i += 1
            if len(content[0]) > 70:
                self.speak_info['Warning'] = '合成文本过长，仅保留前70个音素。'
            spk_id_key = f"{self.speak_info['voice_type']}_{self.speak_info['lang']}_{self.speak_info['emotion']}"
            spk_id = self.speak_id[spk_id_key]
            tts_type = self.vits_model[spk_id_key]
            info = [spk_id, tts_type, len(info4vits)+1, content[0], self.speak_info['rate'],
                    self.speak_info['pitch'], self.speak_info['volume'], break_list, content[1]]
            info4vits.append(info)
            i += 1
        if i == len(merged_sentences)-1:
            content = merged_sentences[i]
            if content[1] == 'silence':
                phonemes_id = info4vits[-1][3]
                prosodies_id = info4vits[-1][-1]
                # print('vitsinfos', info4vits[-1])
                # print('phonemes_id',phonemes_id)
                if phonemes_id[-2] == 7:
                    break_time = break_lists.pop(0)
                    break_time = int(break_time.replace('ms', ''))//10
                    break_list = {len(phonemes_id)-2: break_time}
                elif phonemes_id[-2] == 6:
                    phonemes_id[-2] = 7
                    prosodies_id[-2] = 5
                    break_time = break_lists.pop(0)
                    break_time = int(break_time.replace('ms', ''))//10
                    break_list[len(phonemes_id)-1] = break_time
                else:
                    break_time = break_lists.pop(0)
                    break_time = int(break_time.replace('ms', ''))//10
                    break_list[len(phonemes_id)-1] = break_time
                info4vits[-1][-2] = break_list
            else:
                if len(content[0]) > 70:
                    self.speak_info['Warning'] = '合成文本过长，仅保留前70个音素。'
                break_list = {}
                spk_id_key = f"{self.speak_info['voice_type']}_{self.speak_info['lang']}_{self.speak_info['emotion']}"
                spk_id = self.speak_id[spk_id_key]
                tts_type = self.vits_model[spk_id_key]
                info = [spk_id, tts_type, len(info4vits)+1, content[0], self.speak_info['rate'],
                        self.speak_info['pitch'], self.speak_info['volume'], {}, content[1]]
                info4vits.append(info)
        
        return ' '.join(nor_texts[:-1]), ' '.join(phonemes), ' '.join(prosodies), info4vits, self.speak_info,process_text_memory
    
    
    def process_text_tn(self, text: str, math_type='latex'):

        cleaned_text = self.tp.text_clean(text)

        if math_type:
            detected_text = math_request(
                cleaned_text, self.formular2text_url, type=math_type)
            detected_text = self.tp.remove_spaces_between_cn_en(detected_text)
        else:
            detected_text = cleaned_text

        segments = self.tp.split_by_lang(
            detected_text, lang=self.speak_info['lang'])

        long_sentences = []
        total_tn_time = 0
        for seg in segments:
            text, lang = seg
            if not re.search(r'[a-zA-Z\u4e00-\u9fff0-9]', text):
                long_sentences.append([text, 0])
                continue
            tn_start = time.time()
            nor_text = self.tp.text_normalization(text, lang)
            tn_end = time.time()
            total_tn_time += (tn_end - tn_start)
            if lang == 'cn':
                split_sentences = self.tp.sentence_split_cn(nor_text)
            elif lang == 'en':
                split_sentences = self.tp.sentence_split_en(nor_text)
            long_sentences.extend(split_sentences)
        long_sentences = self.tp.merge_short_sentences(long_sentences)
        return long_sentences
