import requests
import json
import base64
import time
from tqdm import tqdm
# import pandas as pd
def infer_text(api_url, text, voice_params=None, trace_id='test'):
    if voice_params is None:
        # 默认的voice_params
        voice_params = {
            'voice_type': 'xiaosi',
            'emotion': 'sad',
            'lang': 'cn',
            'audio_format': 'wav',
            'rate': '1',
            'pitch': '1',
            'volume': '100'
        }
    
    # 构建请求数据
    request_data = {
        'text': text,
        'voice_params': voice_params,
        'trace_id': trace_id
    }
    
    # 发送 POST 请求到服务器
    start_response = time.time()
    response = requests.post(api_url, json=request_data)
    end_response = time.time()
    print(f"==>> response time: {end_response - start_response}")
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch audio data: {response.status_code}")

    # 解析响应数据
    response_data = response.json()

    base64_data = response_data.get('data')
    if base64_data is None:
        raise Exception(f"Failed to fetch audio data: {response_data}")
    
    data = base64.b64decode(base64_data)
    
    del response_data['data']
    # print(f"==>> response_data: {response_data}")
    # time_log = response_data.get('time_log')
    # print(f"==>> time_log: {time_log}")

    # 保存音频文件
    save_path = f"output_byte_{voice_params['emotion']}.{voice_params['audio_format']}"
    with open(save_path, 'wb') as f:
        print(f"==>> save_path: {save_path}")
        f.write(data)

    end = time.time()
    print(f"Time: {end - start_response}")
    
    return response_data


    
    

if __name__ == '__main__':
    # api_url = "http://localhost:8023/inference"
    api_url = "http://hmi.chengjiukehu.com/tal-vits-pipeline"
    # api_url = "http://genie-internal.vdyoo.net/tal-vits-pipeline/inference"

    # infer_text(api_url,'这道题要我们解决的问题是求一个等腰三角形的顶角的度数，已知它的一个底角的度数为$x°$。')
    # infer_text(api_url,'AAS那全等的目的不就是对应边相等CH等于AO了吗而我们的AO刚才知道长度是三')
    # text = """<speak voice_type=\"xiaosi\" emotion=\"cheerful\" volume=\"90\" rate=\"1.1\" pitch=\"1.0\" lang=\"cn\" audio_format=\"wav\">\n\t北京市8月8日多云，空气湿度46%，\n</speak>"""
    # voice_params={"volume": "", "voice_type": "xiaosi", "emotion": "happy", "rate": "1", "audio_format": "wav", "pitch": "1", "lang": "cn"}
    # voice_params= {"volume": "100", "voice_type": "xiaosi", "emotion": "happy", "rate": "", "audio_format": "wav", "pitch": "1", "lang": "cn"}
    # voice_params= {"volume":"1","voice_type":"xiaosi","emotion":"happy","rate":"1","audio_format":"wav","pitch":"1","lang":"cn"}
    voice_params= {"volume":"100","voice_type":"xiaosi","emotion":"","rate":"1","audio_format":"wav","pitch":"1","lang":"cn"}
    trace_id= "rate427a-db09-4e8f-a808-78bf5a021c4f"
    text="$4.5x+5.5x=(□+□)×□$"
    
    
    for i in range(100):
        response_data = infer_text(api_url,text,voice_params,trace_id)
        emotion = response_data['audio_info']['emotion']
        volume = response_data['audio_info']['volume']
        print(f"==>> emotion: {emotion}, volume: {volume}")
    
    # for i in range(100):
    #     infer_text(api_url,'2024年7月23日，')
    # infer_ssml(api_url)
    

    # with open("QPS_TEST.list",'r') as f:
    #     lines = f.readlines()
    
    # all_result = []
    # for line in tqdm(lines):
    #     print(f"==>> line: {line}")
    #     time_log = infer_text(api_url,line)
    #     # time_log {'process_time': 0.324932336807251, 'tts_time': 0.15348148345947266, 'insert_time': 0.001260995864868164, 'convert_time': 0.0005364418029785156, 'total_pipeline': 0.4802207946777344}
    #     #convert time_log to csv
    #     time_log['text'] = line
    #     time_log['text_len'] = len(line)
    #     all_result.append(time_log)
    # df = pd.DataFrame(all_result)
    # df.to_csv('time_log.csv',index=False)
