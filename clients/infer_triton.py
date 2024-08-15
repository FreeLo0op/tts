import io
import os
import random
import time
import soundfile as sf
import numpy as np
from tqdm import tqdm, trange

from clients.tts_client import TAL_TTS_Triton


class TAL_TTS():
    def __init__(self, url, sample_rate=24000, hop_length=240):
        # self.data_loader_infer = DataLoaderInfer()
        self.tal_tts_triton = TAL_TTS_Triton(url)
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.type_map = {
            "1": "tal_vits",  # 中性
            "286": "tal_vits",  # 疑问
            "283": "tal_vits",  # 开心
            "282": "tal_vits_fear_sad",  # 难过
            "284": "tal_vits_fear_sad",  # 害怕
            "285": "tal_vits_angry", # 生气
            "6": "tal_vits_english"  # 英语
            }  

    def pad_list(self, input_list, target_length=70, padding_value=0):
        """
        Pad the input list to the target length with the padding value.

        Args:
        - input_list (list): The list to be padded.
        - target_length (int): The desired length of the output list. Default is 30.
        - padding_value (any): The value to use for padding. Default is 0.

        Returns:
        - list: The padded list.
        """

        # If the input list is longer than the target length, truncate it
        if len(input_list) > target_length:
            return input_list[:target_length]

        # Otherwise, pad the list with the padding value
        return input_list + [padding_value] * (target_length - len(input_list))

    def infer(self, data):

        target_length = 70

        spk_id, wav_id, text, dur_rate, f0_rate, vol_rate, break_list, rhy_list = data

        input_dict = {}

        text_padding = self.pad_list(text, target_length)
        text_padding_length = len(text_padding)
        text_padding = np.expand_dims(text_padding, axis=0)

        input_dict["text"] = text_padding
        input_dict["text_length"] = np.array(
            [[text_padding_length]], dtype=np.int64)
        # print(f"==>> text_len: {text_len}")
        input_dict["sid"] = np.array([[spk_id]], dtype=np.int64)
        input_dict["dur_rate"] = np.array([[[dur_rate]]], dtype=np.float32)
        input_dict["f0_rate"] = np.array([[f0_rate]], dtype=np.float32)

        rhy_list_padding = self.pad_list(rhy_list, target_length)
        rhy_list = np.expand_dims(rhy_list_padding, axis=0)
        input_dict["rhy"] = np.array(rhy_list, dtype=np.int64)

        ort_outs, duration, f0 = self.tal_tts_triton.get_result(
            input_dict, tts_type=self.type_map[str(spk_id)])

        out_dur = duration[0][0]
        # 去除最后所有为0的元素
        non_zero_index = min(len(text), target_length)  
        # while non_zero_index > 0 and out_dur[non_zero_index - 1] == 0:
        #     non_zero_index -= 1

        # 截取非零部分
        out_dur = out_dur[:non_zero_index]
        duration = out_dur.reshape(1, 1, len(out_dur))

        return (ort_outs, duration)
        f0 = f0[0][:non_zero_index].reshape(1, len(out_dur))

        out_audio = ort_outs[0][0]*(np.array(vol_rate).astype(np.float32))

        # sf.write(r'/mnt/cfs/SPEECH/hupeng/git_loc_workspace/tal_frontend/clients/test.wav', out_audio[:self.hop_length*(int(sum(out_dur)))], self.sample_rate, "PCM_16")

        ort_new = out_audio[:self.hop_length*(int(sum(out_dur)))]

        return (ort_new, duration)
        output_pirnt = {
            "ort_outs": ort_new.tolist(),
            "duration": duration.tolist(),
            "f0": f0.tolist(),
            "hop_length": self.hop_length,
            "format": "WAV",
            "subtype": "PCM_16",
            "time_log": time_log
        }
        # print(f"==>> output_pirnt: {output_pirnt}")

        return output_pirnt

    def run_inference(self, datas):
        all_audio = []
        for data in datas:
            audio = self.infer(data)
            all_audio.append(audio)
        return all_audio


if __name__ == "__main__":
    url = '47.94.0.184:8000'

    tal_tts = TAL_TTS(url, sample_rate=24000, hop_length=240,  output_dir='')

    datas = [['285', 1, [1, 11, 332, 13, 107, 6, 235, 18, 275, 16, 152, 20, 53, 55, 7, 19, 213, 385, 6, 15, 83, 25, 193, 31, 75, 354, 15, 43, 8], '1', '1', '1', {}, [1, 1, 2, 1, 2, 4, 2, 1, 3, 1, 2, 1, 2, 2, 5, 1, 2, 2, 4, 1, 2, 1, 3, 1, 2, 3, 1, 6, 1]], ['285', 2, [1, 31, 85, 12, 305, 333, 9, 99, 6, 31, 42, 30, 185, 6, 26, 63, 12, 45, 25, 192, 23, 98, 6, 31, 62, 16, 234, 18, 213, 12, 86, 6, 315, 385,
                                                                                                                                                                                                                                                                           15, 293, 374, 8], '1', '1', '1', {}, [1, 1, 2, 1, 3, 2, 1, 2, 4, 1, 2, 1, 2, 4, 1, 2, 1, 3, 1, 2, 1, 2, 4, 1, 2, 1, 2, 1, 2, 1, 2, 4, 2, 3, 1, 2, 6, 1]], ['285', 3, [1, 9, 164, 12, 33, 18, 86, 25, 192, 23, 98, 6, 12, 325, 14, 275, 28, 152, 12, 86, 24, 182, 20, 145, 31, 192, 22, 213, 8], '1', '1', '1', {}, [1, 1, 2, 1, 2, 1, 3, 1, 2, 1, 2, 4, 1, 3, 1, 2, 1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 6, 1]]]

    # infer_result = tal_tts.infer(text)
    infer_result = tal_tts.run_inference(datas)
    # del infer_result['ort_outs']
    print(f"==>> time_log: {infer_result}")
