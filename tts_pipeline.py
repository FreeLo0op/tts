import re
import sys
import time
import json

from tal_frontend.tal_tts import Frontend
from post_processing.audio_process import PostProcessing
from clients.infer_triton import TAL_TTS
if __name__ == '__main__':
    fe = Frontend()
    ti = TAL_TTS(url='47.94.0.184:8000')
    pp = PostProcessing()
    
    print('------------------处理纯文本输入示例------------------')
    # 接口参数，在调用process_text时更新，不再在初始化Frontend()时更新
    voice_params = {
        'voice_type': 'xiaosi',
        'emotion': 'happy',
        'lang': 'cn',
        'audio_format': 'mp3',
        'rate': '1',
        'pitch': '1',
        'volume': '100'
    }
    # 指定文本中需要处理的数学公式类型
    math_type = 'latex'
    
    # 返回给用户的json文件
    result = {
        'Error Code':'200',
        'Error Info':'Successed',
        'Text':'',
        'Phoneme':'',
        'Audio':'',
        'Audio Info':''
    }
    # 需要合成的文本
    text = '小思小思，这个数学公式怎么读：$y=ax+b$'
    try:
        # 前端处理
        normalization_text, phonemes, prosodies, inferdatas, speak_info = fe.process_text(text, voice_params=voice_params, math_type=math_type)

        # VITS推理合成音频
        all_audios = ti.run_inference(inferdatas)

        # 后处理，静音段插入和音频格式转换
        res = pp.insert_sil(all_audios, inferdatas)
        res = pp.convert_audio(res, format=speak_info['audio_format'])

        # 保存到本地
        #filename = r'vits/test_audio.{}'.format(speak_info['audio_format'])
        #pp.save_audio(res, filename, format=speak_info['audio_format'])

        result['Text'] = normalization_text
        result['Phoneme'] = phonemes
        result['Audio Info'] = speak_info
        result['prosodie'] = prosodies
    except Exception as e:
        result['Error Code'] = '400'
        result['Error Info'] = e
        result['Text'] = text
    finally:
        print(result)


    sys.exit(0)
    print('------------------处理SSML输入示例------------------')
    # 接口参数，使用ssml可传入改参数，ssml内可以定义音频合成所需要的参数
    voice_params = {
        'voice_type': 'xiaosi',
        'emotion': 'neutral',
        'lang': 'cn',
        'audio_format': 'mp3',
        'rate': '1',
        'pitch': '1',
        'volume': '2'
    }
    # 返回给用户的json文件
    result = {
        'Error Code':'200',
        'Error Info':'Successed',
        'Text':'',
        'Phoneme':'',
        'Audio':'',
        'Audio Info':''
    }
    ssml_file = r'/mnt/cfs/SPEECH/hupeng/git_loc_workspace/tal_frontend/tal_frontend/frontend/ssml/test_ssml_2.xml'    
    try:
        # 前端处理
        normalization_text, phonemes, prosodies, inferdatas, speak_info = fe.process_ssml(ssml_file)
        
        # VITS推理合成音频
        all_audios = ti.run_inference(inferdatas)
        
        # 后处理，静音段插入和音频格式转换
        res = pp.insert_sil(all_audios, inferdatas)
        res = pp.convert_audio(res, format=speak_info['audio_format'])
        
        # 保存到本地
        #filename = r'vits/demo_wavs/ssml_audio_2_1.{}'.format(speak_info['audio_format'])
        #pp.save_audio(res, filename, format=speak_info['audio_format'])

        result['Text'] = normalization_text
        result['Phoneme'] = phonemes
        result['Audio Info'] = speak_info
        result['prosodie'] = prosodies
    except Exception as e:
        result['Error Code'] = '400'
        result['Error Info'] = e
    finally:
        print(result)

