import re
import onnxruntime
import numpy as np
from transformers import BertTokenizer

delete_pattern = re.compile(r'[^\u4e00-\u9fff，。；：、！？a-zA-Z\.\,\?\:\!\;#\d]')
oc_pattern = re.compile(r'(&+)')
en_pattern = re.compile(r'[a-zA-Z]')


def pre_process(text):
    text = delete_pattern.sub(r'', text)
    text = en_pattern.sub(r'&',text)
    text = oc_pattern.sub("&",text).strip()
    return text

def predict(onnx_session, ids, turnoff_tqdm=False):

    probs = onnx_session.run(
        [],
        {
            'input_ids': np.array(ids)
        }
    )[0]

    preds = np.argmax(probs, axis=-1)

    return preds

providers = ['CPUExecutionProvider']
sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
sess_options.intra_op_num_threads = 2
session_g2pw = onnxruntime.InferenceSession("/mnt/cfs/SPEECH/hupeng/git_loc_workspace/tal_frontend_service/tal_frontend/frontend/bertpp/pp.onnx", sess_options=sess_options,providers=providers)
tokenizer = BertTokenizer.from_pretrained("/mnt/cfs/NLP/hub_models/bert-base-chinese")

sentences = ["你好啊，我的朋友"]
token_ids = [tokenizer(sentences[0])["input_ids"]]

print("start")
preds = predict(session_g2pw, token_ids)
preds = preds[0][1:-1]
print(preds)