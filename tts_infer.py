import re
import time
import redis
import yaml
from tqdm import trange, tqdm
from transformers import BertTokenizer

from tal_frontend.tal_tts import Frontend
from post_processing.audio_process import PostProcessing
from clients.infer_triton_batch import TAL_TTS_Batch
from tal_frontend.frontend.g2p_pp.g2p_pp_client import TAL_G2PPP_Triton
from tal_frontend.frontend.g2p.tone_sandhi import ToneSandhi
from tal_frontend.frontend.g2p.zh_frontend import Frontend as zhFrontend
from tal_frontend.frontend.g2p.en_frontend import English
from tal_frontend.frontend.normalizer.cn.cn_normalizer import cn_Normalizer as cn_Normalizer

class TTS_Text_Infer:
    def __init__(self, 
                 g2p_pp_url:str,
                 g2ppp_model_name:str,
                 formular2text_url:str, 
                 tts_triton_url:str, 
                 senlen:int, 
                 en_phoneme_ave_len:int,
                 tokenizer,
                 labels, 
                 char2phonemes,
                 en_monophone,
                 tonesandhi,
                 zh_front,
                 en_frontend,
                 cn_normalizer,
                 redis_client):
        
        # initinal
        self.fe = Frontend(
                                cn_normalizer,
                                formular2text_url, 
                                senlen, 
                                en_phoneme_ave_len)
        self.ti = TAL_TTS_Batch(url=tts_triton_url,
                                g2ppp_model_name=g2ppp_model_name)
        self.pp = PostProcessing()
        
        self.g2p_pp = TAL_G2PPP_Triton(g2p_pp_url,
                                       g2ppp_model_name,
                                       tokenizer,
                                       labels, 
                                       char2phonemes,
                                       en_monophone,
                                       tonesandhi,
                                       zh_front,
                                       en_frontend,
                                       redis_client)
        
      
    
    def infer(self, text, voice_params):
        math_type = 'latex'
        result = {}
        # try:
            # 前端处理
        start = time.time()
        if re.search(r'(<speak.*?>)(.*?)(</speak>)', text, re.DOTALL):
            # print('-------ssml------')
            normalization_text, phonemes, prosodies, inferdatas, speak_info,process_text_memory = self.fe.process_ssml2(text, self.g2p_pp, type='string')
        else:
            # print('-------text------')
            normalization_text, phonemes, prosodies, inferdatas, speak_info,process_text_memory = self.fe.process_text(
                text, self.g2p_pp, voice_params, math_type)
        # return normalization_text
        end_process = time.time()
        # print(f"==>> process_text_memory: {process_text_memory}")

        # print(f"==>> normalization_text: {normalization_text}")
        # print("==>> inferdatas: ",inferdatas)
        # VITS推理合成音频
        # print('inferdatas:', inferdatas)
        all_audios = self.ti.run_inference(inferdatas)
        
        end_tts_time = time.time()
        # 后处理，静音段插入和音频格式转换
        res, durs = self.pp.insert_sil(all_audios, inferdatas)
        end_insert_time = time.time()
        res = self.pp.convert_audio(res, format=speak_info['audio_format'])
        
        end_convert_process = time.time()
        
        result['error_code'] = '200'
        result['data'] = res
        result['text'] = normalization_text
        result['pre_process_text'] = process_text_memory
        result['phoneme'] = phonemes
        result['durations'] = durs
        result['audio_info'] = speak_info
        result['prosody'] = prosodies
        result['inferdatas'] = inferdatas
        result['time_log'] = {"process_time": end_process - start, "tts_time": end_tts_time - end_process, "insert_time": end_insert_time - end_tts_time, "convert_time": end_convert_process - end_tts_time, "total_infer": end_convert_process - start}
        return result

        # except Exception as e:
        #    print(f"==>> Error: {e}")
        #    result['error_code'] = '500'
        #    result['error_info'] = e
        #    result['text'] = text

    def infer_g2p_rhy(self, text, voice_params):
        math_type = 'latex'
        result = {}
        try:
            # 前端处理
            start = time.time()
            if re.search(r'(<speak.*?>)(.*?)(</speak>)', text, re.DOTALL):
                print('-------ssml------')
                normalization_text, phonemes, prosodies, inferdatas, speak_info,process_text_memory = self.fe.process_ssml2(text, self.g2p_pp, type='string')
            else:
                print('-------text------')
                normalization_text, phonemes, prosodies, inferdatas, speak_info,process_text_memory = self.fe.process_text(
                    text, self.g2p_pp, voice_params=voice_params, math_type=math_type)
            #print(f"==>> normalization_text: {normalization_text}")
            process_end = time.time()
            result = {}
            result['text'] = normalization_text
            result['process_text_memory'] = process_text_memory
            result['phoneme'] = phonemes
            result['audio_info'] = speak_info
            result['prosody'] = prosodies
            result['time_log'] = {"process_time": process_end - start}
            result['inferdatas'] = inferdatas
            result['error_code'] = '200'
            return result

        except Exception as e:

           print(f"==>> Error: {e}")

           result['error_code'] = '500'
           result['error_info'] = e
           result['text'] = text

class Config:
    def __init__(self, config_path='config/version.yaml'):
        self.config_path = config_path
        self._load_config()

    def _load_config(self):
        with open(self.config_path) as f:
            yaml_data = yaml.safe_load(f)
        
        self.formular2text_url = yaml_data.get('formular2text_url')
        self.tts_triton_url = yaml_data.get('tts_triton_url')
        self.g2p_pp_url = yaml_data.get('g2p_pp_url')
        
        self.senlen = yaml_data.get('senlen')
        self.en_phoneme_ave_len = yaml_data.get('en_phoneme_ave_len')
        self.bert_model_path = yaml_data.get('bert_model_path')
        self.en_monophone_file = yaml_data.get('en_monophone_file')
        self.polyphonic_chars_path = yaml_data.get('polyphonic_chars_path')
        self.g2ppp_model_name = yaml_data.get('g2ppp_model_name')
        self.redis_client_host = yaml_data.get('host')
        self.redis_client_password = yaml_data.get('password')
        self.redis_client_port = yaml_data.get('port')
        
        
class GetLabels:
    def __init__(self):
        self.config = Config()
        bert_model_path= self.config.bert_model_path
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_path)
        polyphonic_chars_path = self.config.polyphonic_chars_path
        polyphonic_chars = [line.split('\t') for line in open(polyphonic_chars_path).read().strip().split('\n')]

        self.en_monophone = {}
        with open(self.config.en_monophone_file, 'r', encoding='utf8') as fin:
            for line in fin:
                key, value = line.strip().split('\t', maxsplit=1)
                self.en_monophone[key] = value
                
        self.labels, self.char2phonemes = self.get_phoneme_labels(polyphonic_chars)

        self.tonesandhi = ToneSandhi()

    def get_phoneme_labels(self):
        labels = sorted(list(set([phoneme for _, phoneme in self.polyphonic_chars])))
        char2phonemes = {}
        for char, phoneme in self.polyphonic_chars:
            if char not in char2phonemes:
                char2phonemes[char] = []
            char2phonemes[char].append(labels.index(phoneme))
        return labels, char2phonemes  
    
    def get_all(self):
        return self.tokenizer, self.labels, self.char2phonemes, self.en_monophone, self.tonesandhi
    
def main():
    print('------------------Tal TTS Service------------------')
    config = Config()

    tokenizer = BertTokenizer.from_pretrained(config.bert_model_path)
    polyphonic_chars = [line.split('\t') for line in open(config.polyphonic_chars_path).read().strip().split('\n')]

    en_monophone = {}
    with open(config.en_monophone_file, 'r', encoding='utf8') as fin:
        for line in fin:
            key, value = line.strip().split('\t', maxsplit=1)
            en_monophone[key] = value
            
    def get_phoneme_labels( polyphonic_chars):
        labels = sorted(list(set([phoneme for char, phoneme in polyphonic_chars])))
        char2phonemes = {}
        for char, phoneme in polyphonic_chars:
            if char not in char2phonemes:
                char2phonemes[char] = []
            char2phonemes[char].append(labels.index(phoneme))
        return labels, char2phonemes
            
    labels, char2phonemes = get_phoneme_labels(polyphonic_chars)

    tonesandhi = ToneSandhi()
    zh_front, en_frontend = zhFrontend(), English()
    redis_client = None
    # redis_client = redis.Redis(
    #     host=config.redis_client_host,
    #     port=config.redis_client_port,
    #     password=config.redis_client_password,
    #     db = 8
    # )
    cn_normalizer = cn_Normalizer()
    
    tts_infer = TTS_Text_Infer(config.g2p_pp_url,
                               config.g2ppp_model_name,
                               config.formular2text_url,
                               config.tts_triton_url, 
                               config.senlen, 
                               config.en_phoneme_ave_len,
                               tokenizer,
                               labels,
                               char2phonemes,
                               en_monophone,
                               tonesandhi,
                               zh_front,
                               en_frontend,
                               cn_normalizer,
                               None)
    
    # zijie_map = {}
    # with open(r'/mnt/cfs/SPEECH/data/tts/zijie_tts/query_data/20240801_17w/zijie_g2p.txt','r',encoding='utf8') as fin:
    #     lines = fin.readlines()
    #     for i in range(0, len(lines), 2):
    #         line = lines[i]
    #         try:
    #             utt, text = line.strip().split('\t', maxsplit=1)
    #             zijie_map[utt] = text
    #         except:
    #             pass

    # fo = open(r'/mnt/cfs/SPEECH/hupeng/git_loc_workspace/tal_frontend_service/test_data/text/3kw_query.csv','w',encoding='utf8')
    # lines = open(r'/mnt/cfs/SPEECH/data/tts/zijie_tts/query_data/20240801_17w/uniq_3kw_all_data.list','r',encoding='utf8').readlines()

    # for line in tqdm(lines):
    #     try:
    #         utt, pattern, text, _ = line.strip().split('\t')
    #         zijie_text = zijie_map[utt]
    #         input_request = {
    #             'text': text,
    #             'voice_params':{
    #                 'voice_type': 'xiaosi',
    #                 'emotion': 'cheerful',
    #                 'lang': 'cn',
    #                 'audio_format': 'wav',
    #                 'rate': '1',
    #                 'pitch': '1',
    #                 'volume': '100'
    #             }
    #         }

    #         result = tts_infer.infer(input_request['text'], input_request['voice_params'])
            
    #         fo.write(f'{utt}\t{text}\t{result}\t{zijie_text}\n')
    #         fo.flush()
    #     except:
    #         pass
    # fo.close()

    texts = [
        '$$\\frac{1}{2}x^{3}$$', 
        '<speak voice_type=\"xiaosi\" emotion=\"sad\" volume=\"100\" rate=\"1.1\" pitch=\"1.0\"><break time="1000ms"/> \n你好啊,<break time="1000ms"/><break time="1000ms"/>今天天气真差！<break time="1000ms"/><break time="1000ms"/><break time="1000ms"/><break time="1000ms"/></speak>']

    for text in texts[:1]:
        input_request = {
                'text': text,
                'voice_params':{
                    'voice_type': 'xiaosi',
                    'emotion': 'neutral',
                    'lang': 'cn',
                    'audio_format': 'wav',
                    'rate': '1',
                    'pitch': '1',
                    'volume': '100'
                }
            }
        result = tts_infer.infer(input_request['text'], input_request['voice_params'])
        if result:
            print(f"==>> result: {result['text']}\n{result['phoneme']}")
            byte_data = result['data']
            with open('test_data/audio/vits_test.wav', 'wb') as f:
                f.write(byte_data)
            del result['data']
            print(result)

if __name__ == '__main__':
    
    main()
