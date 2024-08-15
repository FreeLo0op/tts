# TTS pipeline
## 1.前端
处理纯文本或SSML，为VITS做数据准备。
返回结果
```
normalization_text:正则化文本
phonemes：音素序列
prosodies：韵律序列
inferdatas：声学模型推理数据
        [['1', 'tal_vits', 1, [1, 24, 185, 12, 144, 274, 455, 421, 427, 16, 132, 273, 12, 144, 274, 455, 421, 427, 6, 12, 109, 363, 31, 110, 13, 62, 28, 213, 16, 132, 31, 110, 13, 62, 28, 213, 12, 86, 15, 83, 6, 11, 108, 124, 31, 110, 13, 62, 28, 213, 8], '1', '1', '100', {}, [1, 1, 2, 1, 3, 2, 1, 1, 3, 1, 2, 2, 1, 3, 2, 1, 1, 4, 1, 1, 2, 3, 1, 2, 1, 2, 1, 3, 1, 3, 1, 2, 1, 2, 1, 2, 1, 3, 1, 4, 1, 1, 2, 3, 1, 2, 1, 2, 1, 6, 1]]]
        顺序：控制声学模型情感的id；声学模型；短句index（长句可能划分为n个短句，目前只做记录，无实际作用）；音素id；rate；pitch；volume；break_list；韵律id。
        
speak_info：用户请求参数 or 默认请求参数
process_text_memory：中间步骤处理结果
```
## 2.VITS
接收前端音素、音色等输出，合成音频。  
改用triton
## 3.后处理
添加用户定义的停顿、拼接短音频为长音频、修改音频输出格式。
## Demo
```py
python tts_infer.py

#所有参数通过input_request传入，
#input_request = {
        #'text': '<speak voice_type=\"xiaosi\" emotion=\"happy\" volume=\"0.9\" rate=\"1.1\" pitch=\"1.0\"> \n你好啊,今天天气</speak>',
#        'text': '小思小思，你好呀，我好喜欢你呀！我们做朋友吧！',
        
#        'voice_params':{
#            'voice_type': 'xiaosi',
#            'emotion': 'neutral',
#            'lang': 'cn',
#            'audio_format': 'wav',
#            'rate': '1',
#            'pitch': '1',
#            'volume': '1'
#        }
#    }
```

## 更新：
1、静音段处理更新，首尾插入静音不在插入#3，直接修改sil和sp4的停顿时间。  
2、后处理保存mp3格式时 bitrate='64k' 保证采样率正确  
3、前端增加音素长度检查，大于70，将warning信息写入返回给用户的json文件  
4、返回json新增韵律 


## Installation
### 现有conda环境
```bash
source /mnt/cfs/SPEECH/hupeng/tools/others_env/hp_env.sh
conda activate py310
```

### 创建conda环境
暂不启用 后续更新
请确保现有环境中没有重名环境py310,如有请自行修改yml文件
```bash
conda env create -f environment.yml
conda activate py310
```

