import os
import numpy as np
import onnxruntime

class Infer:
    def __init__(self, 
                 sample_rate: float = 24000, 
                 hop_length: float = 240,
                 output_format: str = 'wav', 
                 channels: int = 1,
                 bit_depth: int = 16,
                 output_dir: str = 'out_wav_test'):
        
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.output_format = output_format
        self.channels = channels
        self.bit_depth = bit_depth
        #self.output_dir = output_dir
        #os.makedirs(self.output_dir, exist_ok=True)
        self.providers = ["CUDAExecutionProvider"]
        #self.ort_session = onnxruntime.InferenceSession(onnx_model, providers=self.providers)

    def infer(self, data):
        spk_id, wav_id, text, rate, pitch, volume, break_list, prosodies_id = data
        if spk_id in ['1','286','283']:
            onnx_model = r'/mnt/cfs/SPEECH/hupeng/git_loc_workspace/tal_frontend/vits/onnx_models/xiaosi_rhy.onnx'
        elif spk_id in ['282', '284']:
            onnx_model = r'/mnt/cfs/SPEECH/hupeng/git_loc_workspace/tal_frontend/vits/onnx_models/xiaosi_fear_sad_rhy.onnx'
        elif spk_id in ['285']:
            onnx_model = r'/mnt/cfs/SPEECH/hupeng/git_loc_workspace/tal_frontend/vits/onnx_models/xiaosi_angry_rhy.onnx'
        ort_session = onnxruntime.InferenceSession(onnx_model, providers=self.providers)
        ort_inputs = {
            ort_session.get_inputs()[0].name: np.array([text], dtype=np.int64),
            ort_session.get_inputs()[1].name: np.array([len(text)]),
            ort_session.get_inputs()[2].name: np.array([spk_id], dtype=np.int64),
            ort_session.get_inputs()[3].name: np.array([[[rate]]], dtype=np.float32),
            ort_session.get_inputs()[4].name: np.array([[pitch]], dtype=np.float32),
            ort_session.get_inputs()[5].name: np.array([prosodies_id], dtype=np.int64),
        }

        ort_outs, duration, f0 = ort_session.run(None, ort_inputs)
        return (ort_outs, duration)

    def run_inference(self, datas):
        all_audio = []
        for data in datas:
            audio = self.infer(data)
            all_audio.append(audio)

        return all_audio
        
def main():
    datas = [
        ['283', 1, [1, 7, 419, 442, 422, 470, 419, 449, 437, 426, 433, 479, 435, 449, 427, 467, 8], '1', '1', '1', {1: 10}, [1, 5, 1, 1, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 6, 1]], 
        ['283', 3, [1, 125, 15, 323, 406, 11, 192, 8], '1', '1', '1', {}, [1, 3, 1, 3, 3, 1, 6, 1]], 
        ['283', 4, [1, 25, 195, 25, 88, 28, 174, 12, 86, 20, 86, 7, 28, 135, 19, 145, 25, 195, 6, 422, 461, 429, 456, 421, 427, 8], '1', '1', '1', {}, [1, 1, 2, 1, 3, 1, 2, 1, 2, 1, 2, 5, 1, 2, 1, 3, 1, 3, 2, 1, 1, 1, 1, 1, 6, 1]], 
        ['283', 5, [1, 14, 83, 25, 196, 12, 86, 25, 275, 28, 383, 14, 252, 25, 195, 7, 24, 52, 16, 164, 28, 213, 6, 470, 427, 467, 455, 423, 8], '1', '1', '1', {}, [1, 1, 2, 1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 2, 5, 1, 2, 1, 2, 1, 2, 4, 3, 1, 3, 1, 6, 1]], 
        ['283', 6, [1, 16, 205, 24, 185, 12, 109, 363, 6, 24, 52, 16, 164, 28, 213, 6, 442, 413, 467, 467, 7, 28, 135, 19, 145, 25, 195, 8], '1', '1', '1', {}, [1, 1, 2, 1, 3, 1, 2, 2, 4, 1, 2, 1, 2, 1, 3, 2, 4, 1, 3, 2, 5, 1, 2, 1, 2, 1, 6, 1]], 
        ['283', 7, [1, 442, 427, 464, 414, 442, 423, 7, 14, 83, 25, 196, 12, 86, 25, 275, 28, 383, 14, 252, 25, 195, 8], '1', '1', '1', {}, [1, 1, 1, 1, 1, 1, 2, 5, 1, 2, 1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 6, 1]], 
        ['283', 8, [1, 7, 10, 85, 25, 195, 333, 9, 99, 7, 442, 455, 421, 427, 6, 12, 86, 21, 213, 13, 62, 16, 132, 7, 413, 467, 455, 421, 427, 8], '1', '1', '1', {1: 500}, [1, 5, 1, 2, 1, 3, 2, 1, 2, 5, 3, 1, 1, 2, 2, 1, 4, 1, 2, 1, 3, 1, 2, 5, 1, 3, 1, 1, 6, 1]], 
        ['283', 9, [1, 16, 132, 6, 427, 467, 12, 109, 363, 18, 213, 7, 28, 135, 19, 145, 6, 25, 195, 6, 423, 440, 430, 423, 442, 422, 8], '1', '1', '1', {}, [1, 1, 3, 2, 1, 4, 1, 2, 2, 1, 2, 5, 1, 2, 1, 2, 4, 1, 3, 2, 1, 1, 1, 1, 1, 6, 1]], 
        ['283', 10, [1, 14, 83, 25, 196, 12, 86, 25, 275, 28, 383, 14, 252, 25, 195, 8], '1', '1', '1', {}, [1, 1, 2, 1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 6, 1]], 
        ['283', 11, [1, 7, 442, 455, 421, 427, 6, 12, 86, 21, 213, 13, 62, 6, 16, 132, 6, 413, 467, 455, 421, 427, 6, 16, 132, 6, 427, 467, 8], '1', '1', '1', {1: 20}, [1, 5, 3, 1, 1, 2, 2, 1, 4, 1, 2, 1, 2, 4, 1, 3, 2, 1, 4, 1, 1, 2, 4, 1, 2, 2, 1, 6, 1]], 
        ['283', 12, [1, 12, 109, 363, 18, 213, 8], '1', '1', '1', {}, [1, 1, 2, 2, 1, 6, 1]]
             ]

    infer = Infer()
    final_audio = infer.run_inference(datas)
    
if __name__ == "__main__":
    main()
