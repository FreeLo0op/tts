import requests
import time

def fetch_and_infer_text(api_url, text, voice_params=None, trace_id='test'):
    if voice_params is None:
        # 默认的voice_params
        voice_params = {
            'voice_type': 'xiaosi',
            'emotion': 'happy',
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

    print(f"==>> api_url: {api_url}")
    print('------------------处理纯文本输入------------------')
    
    # 发送 POST 请求到服务器
    start_response = time.time()
    response = requests.post(api_url, json=request_data)
    end_response = time.time()
    print(f"==>> response time: {end_response - start_response}")
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch audio data: {response.status_code}")

    # 解析响应数据
    response_data = response.json()
    
    time_log = response_data.get('time_log')
    
    print(f"==>> data: {response_data}")
    print(f"==>> time_log: {time_log}")
    
    end = time.time()
    
    return response_data, time_log

# 示例调用
api_url = "http://localhost:8023/infer_g2p_rhy"
# api_url = "http://hmi.chengjiukehu.com/tal-vits-pipeline/infer_g2p_rhy"
text = "示例文本"
data, time_log = fetch_and_infer_text(api_url, text)
print(data)
print(time_log)