import base64
import random
import time
from locust import HttpUser, TaskSet, task, between
import requests


    

def fetch_audio_from_text(api_url, text, voice_params, trace_id):
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
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch audio data: {response.status_code}")

    # 解析响应数据
    response_data = response.json()

    base64_data = response_data.get('data')
    if base64_data is None:
        raise Exception(f"Failed to fetch audio data: {response_data}")
    data = base64.b64decode(base64_data)
    time_log = response_data.get('time_log')


    return data, time_log



class AudioTaskSet(TaskSet):
    def on_start(self):
        # 读取 JSON 文件
        with open('/mnt/cfs/CV/lj/code/asr_project/tal_frontend/press/tts/text_1w_new.txt') as f:
            self.texts = f.readlines() 
    
    @task
    def fetch_audio(self):
        text = random.choice(self.texts)
        voice_params = {
            'voice_type': 'xiaosi',
            'emotion': 'happy',
            'lang': 'cn',
            'audio_format': 'wav',
            'rate': '1',
            'pitch': '1',
            'volume': '100'
        }
        
        try:
            data, time_log = fetch_audio_from_text('http://hmi.chengjiukehu.com/tal-vits-pipeline', text, voice_params, 'test')
            print(f"Fetched audio data length: {len(data)}")
        except Exception as e:
            print(f"Failed to fetch audio data: {e}")
            
class AudioUser(HttpUser):
    tasks = [AudioTaskSet]
    # wait_time = between(1, 2)
    

if __name__ == "__main__":
    import os
    os.system("locust -f locustfile.py")
    
    
    
   
     