
#先安装客户端
#pip install tritonclient\[all\]
import re
import numpy as np
from transformers import BertTokenizer
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException



class TAL_RHY_Triton():
    def __init__(self,url):
        self.url = url
        self.model_name = "vits_rhy"
        
        self.delete_pattern = re.compile(r'[^\u4e00-\u9fff，。；：、！？a-zA-Z\.\,\?\:\!\;#\d]')
        self.oc_pattern = re.compile(r'(&+)')
        self.en_pattern = re.compile(r'[a-zA-Z]')
        
        self.tokenizer = BertTokenizer.from_pretrained("/mnt/cfs/NLP/hub_models/bert-base-chinese")
        
    def pre_process(self, text):
        #text = self.delete_pattern.sub(r'', text)
        #text = self.en_pattern.sub(r'&',text)
        #text = self.oc_pattern.sub("&",text).strip()
        text = re.sub(r'[a-z]+[\']?[a-z]+|[a-z]|[\,\.\?\:\!，。；：、！？]', '&', text)
        return text

    def post_process(self, rhys:list, words:list, phonemes:list):
        id2rhy = {1:'#0', 2:'#1', 3:'#2', 4:'#3'}
        rhy_res = ['#0']
        phonemes_res = []
        for word, rhy in zip(words, rhys):
            if re.match(r'[\,\.\?\:\!，。；：、！？]', word):
                rhy_res[-1] = '#3'
                rhy_res.append('0')
                #tmp_rhy = ['#3']
                phonemes_res.append('sp3')
            elif rhy != 0:
                phoneme = phonemes.pop(0)
                phoneme = phoneme.split()
                tmp_rhy = ['0'] * (len(phoneme))
                tmp_rhy[-1] = id2rhy[rhy]
                rhy_res.extend(tmp_rhy)
                phonemes_res.extend(phoneme)
            else:
                if rhy_res[-1] == '#0':
                    rhy_res[-1] = '#1'
                phoneme = phonemes.pop(0)
                phoneme = phoneme.split()
                tmp_rhy = ['0'] * (len(phoneme))
                tmp_rhy[-1] = '#1'
                rhy_res.extend(tmp_rhy)
                phonemes_res.extend(phoneme)
        rhy_res = rhy_res[1:]
        for i in range(len(phonemes_res)):
            if phonemes_res[i] == '#3':
                phonemes_res[i] = 'sp3'
                rhy_res[i] = '0'
                try:
                    rhy_res[i-1] = '#3'
                except:
                    pass
        return rhy_res, phonemes_res
    
    def infer(self, sentences, phonemes:list):
        words = re.findall(r'[\u4e00-\u9fff]|[a-z]+[\']?[a-z]+|[a-z]|[\,\.\?\:\!，。；：、！？]', sentences[0])
        #assert len(sen) == len(phonemes)
        sentences = [self.pre_process(i) for i in sentences]
        #sentences = ['一行白鹭上青天& &,你世界七']
        input_ids = [self.tokenizer(sentences[0])['input_ids']]
        input_ids = np.array(input_ids)
        
        inputs = [
        httpclient.InferInput("input_ids", input_ids.shape, "INT64")
    ]

        # Set input data
        inputs[0].set_data_from_numpy(input_ids)
        
    # Create output tensors to receive the results of inference
        outputs = [
        httpclient.InferRequestedOutput("probs"),

    ]
        # Perform inference
        try:
            with  httpclient.InferenceServerClient(url=self.url, verbose=False) as  triton_client:
                results = triton_client.infer(model_name=self.model_name,
                                            inputs=inputs,
                                            outputs=outputs,
                                            headers={"Connection": "close"})
        except InferenceServerException as e:
            print(f"Inference failed: {e}")
            exit(1)

        # Get the output arrays from the results
        probs = results.as_numpy("probs")
        pred_rhys = np.argmax(probs, axis=-1)[0][1:-1].tolist()
        rhy_res, phonemes_res = self.post_process(pred_rhys, words, phonemes)
        return rhy_res, phonemes_res
    
if __name__ == "__main__":

    '''
    标点符号 -> 0 -> 变成#3
    英文 -> & -> 0 根据规则加韵律
    1 -> #0
    2 -> #1
    3 -> #2
    4 -> #3
    英文加韵律规则: 韵律处理方法为在空格后加#1，中英文交接处加#1，标点符号后加#3。
    
    单个字的韵律映射到音素 比如
    音节 韵律  ===>   音节  韵律   
    'ni3'  1    ===>  [n, i3]  [0, #0]
    
    'HH AH0 L OW1' 0  ==> [HH, AH0, L, OW1]  [0, 0, 0, #1] 
    
    'ni3 hao3 HH EH0 L OW1。'  1 2 0 0 ==>  ['n', 'i2', 'h', 'ao3', 'HH', 'EH0', 'L', 'OW1', 'sp3'] ['0', '#0', '0', '#1', '0', '0', '0', '#1', '#3']
    
    rhy_id_map = {'_': 0, '0': 1, '#0': 2, '#1': 3, '#2': 4, '#3': 5, '#4': 6}
    '''
    url = "123.56.235.205:80"
    
    client = TAL_RHY_Triton(url)
    
    # Example input data (replace this with actual data)
    # input_ids = np.array([1, 2, 3, 4], dtype=np.int64).reshape(1, -1)
    input_ids_batch = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int64)  # Batch of 2 sequences
    phonemes = ['y i4', 'x ing2', 'b ai2', 'l u4', 'sh ang4', 'q ing1', 't ian1', 'IH1 T', 'M IY1', 'n i3', 'sh iii4', 'j ie4', 'q i1']
    sentences = ["一行白鹭上青天。it's me,你世界七"]
    sentences = ["rember to be happy!"]
    probs = client.infer(sentences, phonemes)
    print("Inference result:", probs, len(probs))
    
    
