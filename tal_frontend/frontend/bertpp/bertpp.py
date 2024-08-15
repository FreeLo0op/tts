import re
import onnxruntime
import numpy as np
from transformers import BertTokenizer

class PP_INFER:
    
    def __init__(self) -> None:
        
        self.delete_pattern = re.compile(r'[^\u4e00-\u9fff，。；：、！？a-zA-Z\.\,\?\:\!\;#\d]')
        self.oc_pattern = re.compile(r'(&+)')
        self.en_pattern = re.compile(r'[a-zA-Z]')
        
        self.providers = ['CPUExecutionProvider']
        self.sess_options = onnxruntime.SessionOptions()
        self.sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        self.sess_options.intra_op_num_threads = 2
        
        self.session_g2pw = onnxruntime.InferenceSession("/mnt/cfs/SPEECH/hupeng/git_loc_workspace/tal_frontend_service/tal_frontend/frontend/bertpp/pp.onnx", sess_options=self.sess_options,providers=self.providers)
        self.tokenizer = BertTokenizer.from_pretrained("/mnt/cfs/NLP/hub_models/bert-base-chinese")
        pass
    
    def pre_process(self, text):
        text = self.delete_pattern.sub(r'', text)
        text = self.en_pattern.sub(r'&',text)
        text = self.oc_pattern.sub("&",text).strip()
        return text

    def pp_predict(self, sentences, turnoff_tqdm=False):
        
        sentences = [self.pre_process(i) for i in sentences]
        
        token_ids = [self.tokenizer(sentences[0])['input_ids']]
        
        probs = self.session_g2pw.run(
            [],
            {
                'input_ids' : np.array(token_ids)
            }
        )[0]
        
        preds = np.argmax(probs, axis=-1)
        
        return preds[0]
        

if __name__ == '__main__':
    pp_infer = PP_INFER()
    
    sentences = ["一行白鹭上青天hello world你世界"]
    
    preds = pp_infer.pp_predict(sentences)
    print(len(preds) , preds)