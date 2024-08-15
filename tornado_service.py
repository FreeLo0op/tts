from tqdm import trange
import time
import redis
import tornado.ioloop
import tornado.web
import json
import base64
import numpy as np
import logging
import yaml
from enum import Enum
from transformers import BertTokenizer
from tal_frontend.frontend.g2p.tone_sandhi import ToneSandhi
from tal_frontend.frontend.g2p.zh_frontend import Frontend as zhFrontend
from tal_frontend.frontend.g2p.en_frontend import English
from tal_frontend.frontend.normalizer.cn.cn_normalizer import cn_Normalizer as cn_Normalizer
from tools.error_config import ErrorCode

from tts_infer import TTS_Text_Infer,Config
from tools.logger import Log

basePath = '/home/logs/xeslog'
 
_PORT = 8023
logger = Log(level=logging.INFO, port=_PORT, debug=False)



# 定义错误代码枚举



class InferenceHandler(tornado.web.RequestHandler):
    def initialize(self,config,tokenizer,labels, char2phonemes,en_monophone,tonesandhi,zh_front,en_frontend ,cn_normalizer):
        # self.tokenizer = tokenizer
        self.config  = config
        self.config = config
        self.tokenizer = tokenizer
        self.labels = labels
        self.char2phonemes = char2phonemes
        self.en_monophone = en_monophone
        self.tonesandhi = tonesandhi
        self.zh_front = zh_front
        self.en_frontend = en_frontend
        self.cn_normalizer = cn_normalizer
        
        
        
        

    def post(self,):
        trace_id = 'unknown'  # 初始化 trace_id
        text = ''  # 初始化 text
        data = json.loads(self.request.body)
        print(f"==>> data: {data}")
        try:
          

            text = data.get('text', '')
            trace_id = data.get('trace_id', 'unknown')
        except Exception as e:
            logger.error(code=ErrorCode.PARAMERTER_ERROR.value, message={"text": text, "error_message": str(e)}, trace_id=trace_id)

            self.set_status(200)
            error_info = {
                "trace_id": trace_id,
                "error_code": ErrorCode.PARAMERTER_ERROR.value,
                "error_message": str(e)}
            self.write(error_info)
            
        
        try:
        #     # 解析请求中的JSON数据
            start_time = time.time()    
            data = json.loads(self.request.body)

            text = data.get('text', '')
            trace_id = data.get('trace_id', 'unknown')

            
        
            voice_params = data.get('voice_params', None)
            print(f"==>> start infer voice_params: {voice_params}")
            DEFAULT_SPEAK_INFO = {
            'voice_type': 'xiaosi',
            'emotion': 'neutral',
            'lang': 'cn',
            'audio_format': 'wav',
            'rate': '1',
            'pitch': '1',
            'volume': '100'
        }
            
            # for k,v in DEFAULT_SPEAK_INFO.items():
            #     if k not in voice_params:
            #         voice_params[k] = v
                
            # for k,v in voice_params.items():
            #     if voice_params[k] is None or voice_params[k] == '':
            #         voice_params[k] = DEFAULT_SPEAK_INFO[k]
            logger.info(code=ErrorCode.SUCCESS.value, message={"tag":"server初始化", "text": text, "voice_params": voice_params,}, trace_id=trace_id)
            # infer_result = self.tal_text_tts.infer(text, voice_params)
            start_init_time = time.time()
            tts_infer = TTS_Text_Infer(self.config.g2p_pp_url,
                                            self.config.g2ppp_model_name, 
                                            self.config.formular2text_url, 
                                            self.config.tts_triton_url, 
                                            self.config.senlen, 
                                            self.config.en_phoneme_ave_len,
                                            self.tokenizer,
                                            self.labels, 
                                            self.char2phonemes,
                                            self.en_monophone,
                                            self.tonesandhi,
                                            self.zh_front,
                                            self.en_frontend,
                                            self.cn_normalizer,
                                            None )

            start_infer = time.time()
            infer_result = tts_infer.infer(text, voice_params)
            end_infer = time.time()
            # infer_result['time_log']['infer_time'] = end_infer - start_infer
            # print(f"==>> infer_time : {end_infer - start_infer}")
            time_log = infer_result.get('time_log')


            # 返回结果作为JSON响应

            infer_result['status_code'] = 200
            infer_result['error_code'] = ErrorCode.SUCCESS.value
            infer_result['trace_id'] = trace_id
            infer_result['time_log'] = time_log
            start_base64 = time.time()
            encoded_data = base64.b64encode(
                infer_result['data']).decode('utf-8')

            infer_result['data'] = encoded_data
            
            # 获取编码数据的大小（以字节为单位）
            encoded_data_size_bytes = len(encoded_data.encode('utf-8'))

            # 将大小转换为千字节（KB）
            encoded_data_size_kb = encoded_data_size_bytes / 1024
            
            end_time = time.time()
            time_log['init_time'] = start_infer - start_init_time
            time_log['base64_time'] = end_time - start_base64
            time_log['encoded_data_size_kb'] =  f"{encoded_data_size_kb:.2f} KB"
            time_log['total_pipeline'] = end_time - start_time

            # logger.info(f"trace_id={trace_id} text={text} time_log={time_log}")
            logger.info(code=ErrorCode.SUCCESS.value, 
                message={
                "request_json": data,

                "text": text,
                "time_log": time_log,
                "pre_process_text": infer_result['pre_process_text'],
                "phoneme": infer_result['phoneme'],
                "durations": infer_result['durations'],
                "audio_info": infer_result['audio_info'],
                "prosody": infer_result['prosody'],
                "normalization_text": infer_result['text'],
                "inferdatas": infer_result['inferdatas']
                },
                 trace_id=trace_id)
            self.write(json.dumps(infer_result))


        except ValueError as e:
            self.set_status(200)  # 设置HTTP状态码为400 Bad Request
            error_info = {
                        "trace_id": trace_id,
                        "request_json": data,

                        "error_code": ErrorCode.INVALID_INPUT.value,
                        "error_message": str(e)}
            # logger.error(f"trace_id={trace_id} || text={text}  ||  error={error_info}  ")
            logger.error(code=ErrorCode.INVALID_INPUT.value, message={
                "text": text,
                "request_json": data,

                "error_message": str(e)
                }, 
                trace_id=trace_id)

            self.write(error_info)


        except Exception as e:
            self.set_status(200)  # 设置HTTP状态码为400 Bad Request
            
            error_info = {
                "trace_id": trace_id,
                "error_code": ErrorCode.INTERNAL_ERROR.value,
                "error_message": str(e)
            }
            
            # logger.error(f"trace_id={trace_id} || text={text}  ||  error={error_info} ")
            logger.error(code=ErrorCode.INVALID_INPUT.value, message={
                "text": text,
                "request_json": data,
                "error_message": str(e)
                }, 
                trace_id=trace_id)
      
            self.write(json.dumps(error_info))
            

class Inference_G2P_RHY_Handler(tornado.web.RequestHandler):
    def initialize(self,config,tokenizer,labels, char2phonemes,en_monophone,tonesandhi,zh_front,en_frontend ,cn_normalizer):
        # self.tokenizer = tokenizer
        
        self.tts_infer = TTS_Text_Infer(config.g2p_pp_url,config.g2ppp_model_name,  config.formular2text_url, config.tts_triton_url, 
                                        config.senlen, config.en_phoneme_ave_len,tokenizer,labels, char2phonemes,
                                        en_monophone,tonesandhi,zh_front,en_frontend ,cn_normalizer,None)



    def post(self,):
        
        
        trace_id = 'unknown'  # 初始化 trace_id
        text = ''  # 初始化 text
        data = json.loads(self.request.body)
        print(f"==>> data: {data}")
        try:
            

            text = data.get('text', '')
            trace_id = data.get('trace_id', 'unknown')
        except Exception as e:
            logger.error(code=ErrorCode.PARAMERTER_ERROR.value, message={"text": text, "error_message": str(e)}, trace_id=trace_id)

            self.set_status(200)
            error_info = {
                "trace_id": trace_id,
                "error_code": ErrorCode.PARAMERTER_ERROR.value,
                "error_message": str(e)}
            self.write(error_info)
            
        try:
        #     # 解析请求中的JSON数据
            start_time = time.time()    
            data = json.loads(self.request.body)
            text = data.get('text', '')
            trace_id = data.get('trace_id', 'unknown')
            voice_params = data.get('voice_params', None)
            print(f"==>> start infer voice_params: {voice_params}")
            # infer_result = self.tal_text_tts.infer(text, voice_params)
            start_infer = time.time()
            infer_result = self.tts_infer.infer_g2p_rhy(text, voice_params)
            end_infer = time.time()
            print(f"==>> infer_time : {end_infer - start_infer}")
          # 返回结果作为JSON响应
            infer_result['status_code'] = 200
            infer_result['trace_id'] = trace_id
           
            end_time = time.time()
            time_log={}
            time_log['total_pipeline'] = end_time - start_time

            # logger.info(f"trace_id={trace_id} || text={text}  ||  result={infer_result}  ||  time_log={time_log}")
            logger.info(code=ErrorCode.SUCCESS.value,
                message={
                "request_json": data,
                "text": text,
                "process_text_memory": infer_result['process_text_memory'],
                "phoneme": infer_result['phoneme'],
                "audio_info": infer_result['audio_info'],
                "prosody": infer_result['prosody'],
                "normalization_text": infer_result['text'],
                "inferdatas": infer_result['inferdatas'],
                "time_log": time_log
        
                },
                 trace_id=trace_id)
            self.write(json.dumps(infer_result))

        except ValueError as e:
            self.set_status(200)  # 设置HTTP状态码为400 Bad Request
            error_info = {
                        "trace_id": trace_id,
                        "request_json": data,
                        "error_code": ErrorCode.INVALID_INPUT.value,
                        "error_message": str(e)}
            logger.error(code=ErrorCode.INVALID_INPUT.value, message={
                "text": text,
                "error_message": str(e)
                }, 
                trace_id=trace_id)

            self.write(error_info)


        except Exception as e:
            self.set_status(200)  # 设置HTTP状态码为400 Bad Request
            error_info = {
                        "trace_id": trace_id,
                        
                        "error_code": ErrorCode.INVALID_INPUT.value,
                        "error_message": str(e)}
            
            logger.error(code=ErrorCode.INVALID_INPUT.value, message={
                "text": text,
                "request_json": data,
                "error_message": str(e)
                }, 
                trace_id=trace_id)
      
            self.write(json.dumps(error_info))

# def set_logging(server_port):
#     formatter = logging.Formatter(
#         '[timestamp=%(asctime)s] [filename=%(filename)s] [level=%(levelname)s]  [app_id=tal_tts] %(message)s')
#     file_handler = logging.handlers.TimedRotatingFileHandler(
#         "/home/logs/xeslog/tal_tts_%d.log" % (server_port), when='D', interval=1, backupCount=7)
#     file_handler.setFormatter(formatter)

#     stdoutHandler = logging.StreamHandler()
#     stdoutHandler.setFormatter(formatter)

#     logger.addHandler(file_handler)
#     logger.addHandler(stdoutHandler)


if __name__ == "__main__":

    
    # config = Config()
    # bert_model_path = config.bert_model_path
    # tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    # print(f"==>> self.bert_model_path: {bert_model_path},loaded success")
    
    # polyphonic_chars_path = "tal_frontend/frontend/g2p_pp/POLYPHONIC_CHARS.txt"
    # polyphonic_chars = [line.split('\t') for line in open(polyphonic_chars_path).read().strip().split('\n')]

    # en_monophone = {}
    # en_monophone_file = r'tal_frontend/frontend/g2p/bertg2pw/english_dict.list'
    # with open(en_monophone_file, 'r', encoding='utf8') as fin:
    #     for line in fin:
    #         key, value = line.strip().split('\t', maxsplit=1)
    #         en_monophone[key] = value
            
    config = Config()

    bert_model_path= config.bert_model_path
    print(f"==>> self.bert_model_path: {bert_model_path}")
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)
    polyphonic_chars = [line.split('\t') for line in open(config.polyphonic_chars_path).read().strip().split('\n')]

    # polyphonic_chars = [line.split('\t') for line in open(polyphonic_chars_path).read().strip().split('\n')]

    en_monophone = {}
    en_monophone_file = r'tal_frontend/frontend/g2p/bertg2pw/english_dict.list'
    with open(en_monophone_file, 'r', encoding='utf8') as fin:
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

    cn_normalizer = cn_Normalizer()
            
    labels, char2phonemes = get_phoneme_labels(polyphonic_chars)

    tonesandhi = ToneSandhi()
    zh_front,en_frontend = zhFrontend(),English()
    
    redis_client = None
    # redis_client = redis.Redis(
    #     host=config.redis_client_host,
    #     port=config.redis_client_port,
    #     password=config.redis_client_password,
    #     db = 8
    # )
    


    app = tornado.web.Application([
        (r"/", InferenceHandler, {"config":config, "tokenizer":tokenizer,"en_monophone":en_monophone,"labels":labels, "char2phonemes":char2phonemes,"tonesandhi":tonesandhi,"zh_front":zh_front,"en_frontend":en_frontend,'cn_normalizer':cn_normalizer}),
        (r"/inference", InferenceHandler, {"config":config, "tokenizer":tokenizer,"en_monophone":en_monophone,"labels":labels, "char2phonemes":char2phonemes,"tonesandhi":tonesandhi,"zh_front":zh_front,"en_frontend":en_frontend,'cn_normalizer':cn_normalizer}),
        (r"/infer_g2p_rhy", Inference_G2P_RHY_Handler, {"config":config, "tokenizer":tokenizer,"en_monophone":en_monophone,"labels":labels, "char2phonemes":char2phonemes,"tonesandhi":tonesandhi,"zh_front":zh_front,"en_frontend":en_frontend,'cn_normalizer':cn_normalizer}),
        
        # (r"/", InferenceHandler),
        # (r"/inference", InferenceHandler),

    ],
        debug=False)
    http_server = tornado.httpserver.HTTPServer(app)
    # http_server.listen(_PORT)

    # set_logging(_PORT)
    # logging.basicConfig(level=logging.INFO)
    # tornado 多进程
    http_server.bind(_PORT)
    http_server.start(0)

    print(f"==>> Starting server on port {_PORT}")

    # tornado.ioloop.IOLoop.current().start()
    tornado.ioloop.IOLoop.instance().start()  # 开始事件