import time
import base64
import json
from locust import HttpUser, TaskSet, task, between

class UserBehavior(TaskSet):

    def fetch_audio_from_text(self, api_url, text, voice_params, trace_id):
        # 构建请求数据
        request_data = {
            'text': text,
            'voice_params': voice_params,
            'trace_id': trace_id
        }

        # 发送 POST 请求到服务器
        with self.client.post(api_url, json=request_data, catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Failed to fetch audio data: {response.status_code}")
                return None
            
            response_data = response.json()

            base64_data = response_data.get('data')
            if base64_data is None:
                response.failure(f"Failed to fetch audio data: {response_data}")
                return None

            data = base64.b64decode(base64_data)
            time_log = response_data.get('time_log')
            print(f"==>> time_log: {time_log}")

            return data, time_log

    def infer_text(self, api_url, text):
        print('------------------处理纯文本输入------------------')
        voice_params = {
            'voice_type': 'xiaosi',
            'emotion': 'neutral',
            'lang': 'cn',
            'audio_format': 'mp3',
            'rate': '1',
            'pitch': '1',
            'volume': '2'
        }
        
        start = time.time()
        trace_id = 'test'
        
        result = self.fetch_audio_from_text(api_url, text, voice_params, trace_id)
        
        end = time.time()
        
        if result is not None:
            byte_data, time_log = result
            print(f"Time: {end - start}")
            return time_log
        
    @task(1)
    def test_infer_text(self):
        api_url = "/tal-vits-pipeline"
        with open("QPS_TEST.list", "r") as f:
            lines = f.readlines()
        
        for line in lines:
            print(f"==>> line: {line.strip()}")
            self.infer_text(api_url, line.strip())

class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(1, 2)  # 每个任务之间的等待时间

if __name__ == '__main__':
    import os
    os.system("locust -f locustfile.py")
