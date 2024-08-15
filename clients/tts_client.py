
#先安装客户端
#pip install tritonclient\[all\]

import numpy as np
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

class TAL_TTS_Triton():
    def __init__(self,url,g2ppp_model_name):
        self.url = url
        self.model_name = g2ppp_model_name
 
    def get_result(self, input_dict,tts_type):
        model_name = tts_type
        
        text = input_dict["text"]
        text_len = input_dict["text_length"]
        spk_id = input_dict["sid"]
        dur_rate = input_dict["dur_rate"]
        f0_rate = input_dict["f0_rate"]
        rhy = input_dict["rhy"]
        inputs = [
        httpclient.InferInput("text", text.shape, "INT32"),
        httpclient.InferInput("text_length", text_len.shape, "INT32"),
        httpclient.InferInput("sid", spk_id.shape, "INT32"),
        httpclient.InferInput("dur_rate", dur_rate.shape, "FP32"),
        httpclient.InferInput("f0_rate", f0_rate.shape, "FP32"),
        httpclient.InferInput("rhy", rhy.shape, "INT32")
    ]
        # Set input data
        
        inputs[0].set_data_from_numpy(text)
        inputs[1].set_data_from_numpy(text_len)
        inputs[2].set_data_from_numpy(spk_id)
        inputs[3].set_data_from_numpy(dur_rate)
        inputs[4].set_data_from_numpy(f0_rate)
        inputs[5].set_data_from_numpy(rhy)

    # Create output tensors to receive the results of inference
        outputs = [
        httpclient.InferRequestedOutput("duration"),
        httpclient.InferRequestedOutput("pitch"),
        httpclient.InferRequestedOutput("audio")
    ]
        # Perform inference
        try:
            with  httpclient.InferenceServerClient(url=self.url, verbose=False) as  triton_client:
                results = triton_client.infer(model_name=model_name,
                                            inputs=inputs,
                                            outputs=outputs,
                                            headers={"Connection": "close"})
        except InferenceServerException as e:
            print(f"Inference failed: {e}")
            exit(1)

        # Get the output arrays from the results
        duration = results.as_numpy("duration")
        pitch = results.as_numpy("pitch")
        audio = results.as_numpy("audio")
        
        # print(f"Duration: {duration}")
        # print(f"Pitch: {pitch}")
        # print(f"Audio: {audio}")
        return audio,duration,pitch
    
if __name__ == "__main__":
    # input_dict = np.array([{"text": "I love you", "text_pair": "I like you"},{"text": "I love you", "text_pair": "I hate you"}])
    # input_dict = np.array([{"text":"I love you", "text_pair":"I like you very"}])

    # bge_client = BGE_client()
    # result = bge_client.get_result(input_dict)
    # print(f"==>> result: {result}")
    
    text = np.array([[1, 22, 202, 19, 32, 364, 15, 265, 19, 32, 7, 12, 325,
                      9, 275, 14, 252, 26, 63, 8],
                     [1, 22, 202, 19, 32, 364, 15, 265, 19, 32, 7, 12, 325,
                      9, 275, 14, 252, 26, 63, 8]], dtype=np.int32)

    text_len = np.array([20, 20], dtype=np.int32)
    text_len = text_len[:, np.newaxis]

    spk_id = np.array([[2], [2]], dtype=np.int32)
    dur_rate = np.array([[[0.8]], [[1.0]]], dtype=np.float32)
    f0_rate = np.array([[1.0], [1.1]], dtype=np.float32)
    
    input_dict = {}
    input_dict["text"] = text
    input_dict["text_length"] = text_len
    input_dict["sid"] = spk_id
    input_dict["dur_rate"] = dur_rate
    input_dict["f0_rate"] = f0_rate
    
    # tal_tts_client = TAL_TTS_Triton("10.171.15.145:8000")
    tal_tts_client = TAL_TTS_Triton("10.171.15.145:8000")
    duration,pitch,audio = tal_tts_client.get_result(input_dict)
    
    
    print(f"Duration: {duration}")
    print(f"Pitch: {pitch}")
    print(f"Audio: {audio}")

    
    
    
