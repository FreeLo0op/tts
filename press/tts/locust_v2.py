import random
from locust import HttpUser, TaskSet, task, between

class UserBehavior(TaskSet):
    def on_start(self):
        # 读取 JSON 文件
        with open('/mnt/cfs/CV/lj/code/asr_project/tal_frontend/press/tts/text_1w_new.txt') as f:
            self.texts = f.readlines() 
    
    @task(1)
    def post_request(self):
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
    
        api_url = "/tal-vits-pipeline/inference"  # 替换为你的API端点
        request_data = {
            'text': text,  # 替换为你的实际文本数据
            'voice_params': voice_params,  # 替换为你的实际voice_params数据
            'trace_id': "example_trace_id"  # 替换为你的实际trace_id
        }
        self.client.post(api_url, json=request_data)

class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    # wait_time = between(1, 5)  # 每个用户在任务之间等待的时间（秒）
    host = "http://hmi.chengjiukehu.com"  # 设置主机地址

if __name__ == "__main__":
    import os
    os.system("locust -f locustfile.py")
