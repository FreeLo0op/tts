import re
import sys
import time
import json
import yaml
from tqdm import trange

from tal_frontend.tal_tts import Frontend
from post_processing.audio_process import PostProcessing
from clients.infer_triton import TAL_TTS
from clients.infer_triton_batch import TAL_TTS_Batch


class TTS_Text_Infer:
    def __init__(self, llm_url, formular2text_url, tts_triton_url):
        self.fe = Frontend(llm_url, formular2text_url)
        # self.ti = TAL_TTS(url=tts_triton_url)
        self.ti = TAL_TTS_Batch(url=tts_triton_url)
        self.pp = PostProcessing()

    def infer(self, text, voice_params):
        

        math_type = 'latex'
  
        result = {}

       
        # 前端处理
        start = time.time()
        if voice_params is  None and text.startswith('<speak'):
            normalization_text, phonemes, prosodies, inferdatas, speak_info = self.fe.process_ssml(text, type='string')

        else:
            normalization_text, phonemes, prosodies, inferdatas, speak_info = self.fe.process_text(
                text, voice_params=voice_params, math_type=math_type)
        print(f"==>> normalization_text: {normalization_text}")

        process_end = time.time()

        # VITS推理合成音频
        all_audios = self.ti.run_inference(inferdatas)
        tts_end = time.time()

        # 后处理，静音段插入和音频格式转换
        res = self.pp.insert_sil(all_audios, inferdatas)
        insert_end = time.time()
        res = self.pp.convert_audio(res, format=speak_info['audio_format'])

        # 保存到本地
        # filename = r'vits/demo_wavs/{}.{}'.format(
        #     key, speak_info['audio_format'])
        # pp.save_audio(res, filename, format=speak_info['audio_format'])

        byte_data = self.pp.convert_audio_tobyte(
            res,  format=speak_info['audio_format'])
        convert_end = time.time()

        time_log = {"process_time": process_end - start,
                    "tts_time": tts_end - process_end,
                    "insert_time": insert_end - tts_end,
                    "convert_time": convert_end - insert_end}
        result['data'] = byte_data
        result['text'] = normalization_text
        result['phoneme'] = phonemes
        result['audio_info'] = speak_info
        result['prosodie'] = prosodies
        result['time_log'] = time_log
        return result

    

class TTS_SSML_Infer:
    def __init__(self, llm_url, formular2text_url, tts_triton_url):
        self.fe = Frontend(llm_url, formular2text_url)
        # self.ti = TAL_TTS(url=tts_triton_url)
        self.ti = TAL_TTS_Batch(url=tts_triton_url)
        self.pp = PostProcessing()

    def infer(self, ssml_file):

        result = {}
        try:
            # 前端处理
            start = time.time()
            normalization_text, phonemes, prosodies, inferdatas, speak_info = self.fe.process_ssml(ssml_file, type='string')
            print(f"==>> normalization_text: {normalization_text}")

            
            process_end = time.time()
            # VITS推理合成音频
            all_audios = self.ti.run_inference(inferdatas)
            tts_end = time.time()

            # 后处理，静音段插入和音频格式转换
            res = self.pp.insert_sil(all_audios, inferdatas)
            insert_end = time.time()

            res = self.pp.convert_audio(res, format=speak_info['audio_format'])

            # 保存到本地
            # filename = r'test_result/ssml_audio_2_1.{}'.format(
            #     speak_info['audio_format'])
            # self.pp.save_audio(
            #     res, filename, format=speak_info['audio_format'])
            byte_data = self.pp.convert_audio_tobyte(
                res,  format=speak_info['audio_format'])
            convert_end = time.time()

            time_log = {"process_time": process_end - start,
                        "tts_time": tts_end - process_end,
                        "insert_time": insert_end - tts_end,
                        "convert_time": convert_end - insert_end}
            result['data'] = byte_data
            result['text'] = normalization_text
            result['phoneme'] = phonemes
            result['audio_info'] = speak_info
            result['time_log'] = time_log
            result['prosodie'] = prosodies
            return result
        except Exception as e:
            result['error_code'] = '400'
            result['error_info'] = e
            print(f"==>> error: {e}")
       


def process_text():
    print('------------------处理纯文本输入------------------')
    # 接口参数，在调用process_text时更新，不再在初始化Frontend()时更新
    voice_params = {
        'voice_type': 'xiaosi',
        'emotion': 'neutral',
        'lang': 'cn',
        'audio_format': 'mp3',
        'rate': '1',
        'pitch': '1',
        'volume': '2'
    }

    # 需要合成的文本
    input = 'xiaosi_neutral_021	这段文本摘自唐代诗人张九龄的《望月怀远》，表达了诗人对故乡的思念之情。xiaosi_neutral_021	这段文本摘自唐代诗人张九龄的《望月怀远》，表达了诗人对故乡的思念之情xiaosi_neutral_021	这段文本摘自唐代诗人张九龄的《望月怀远》，表达了诗人对故乡的思念之情'

    with open('config/version.yaml') as f:
        yaml_data = yaml.safe_load(f)
    llm_url = yaml_data['llm_url']
    formular2text_url = yaml_data['formular2text_url']
    tts_triton_url = yaml_data['tts_triton_url']

    tts_infer = TTS_Text_Infer(llm_url, formular2text_url, tts_triton_url)
    result = tts_infer.infer(input, voice_params)
 
    # for i in trange(100):
    #     start = time.time()
    #     tts_infer.infer(input, voice_params)
    #     end = time.time()
    #     print(f"Time: {end - start}")
    # print(result)
    # byte_data = result['data']
    # with open('output_test.mp3', 'wb') as f:
    #     f.write(byte_data)


def process_ssml():
    with open('config/version.yaml') as f:
        yaml_data = yaml.safe_load(f)
    llm_url = yaml_data['llm_url']
    formular2text_url = yaml_data['formular2text_url']
    tts_triton_url = yaml_data['tts_triton_url']

    tts_infer = TTS_SSML_Infer(llm_url, formular2text_url, tts_triton_url)

    ssml_file = r'/mnt/cfs/CV/lj/code/asr_project/tal_frontend/tal_frontend/frontend/ssml/test_ssml_1.xml'
    with open(ssml_file, 'r') as fin:
        ssml_file = fin.read()
        print(f"==>> ssml_file: {ssml_file}")
    tts_infer.infer(ssml_file)


if __name__ == '__main__':
    process_text()
    # process_ssml()
