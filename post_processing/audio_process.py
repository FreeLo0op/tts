#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# TTS服务音频处理模块
'''
@File : audio_process.py
@Author : HuPeng
@Email : hupeng3@tal.com
'''
import io
import subprocess
import numpy as np
from pydub import AudioSegment

class PostProcessing:
    def __init__(self,
                 sample_rate: float = 24000, 
                 hop_length: float = 240,
                 output_format: str = 'wav', 
                 channels: int = 1,
                 bit_depth: int = 16,
                 ) -> None:
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.output_format = output_format
        self.channels = channels
        self.bit_depth = bit_depth
        
    def insert_sil(self, audios, datainofs):
        '''
        audio: vite推理的结果，元祖类型
        break_list: 插入静音段的位置和时长
        text: 音素id列表
        '''
        out_audio_list = []
        durs = list()
        for audio, datainfo in zip(audios, datainofs):
            ort_outs, duration = audio
            text_list = datainfo[3]
            end_index = len(text_list) - 1
            volume = float(datainfo[6]) / 100
            break_list = datainfo[-2] if datainfo[-2] else {0:10, end_index:10}
            
            for out_dur, out_audio in zip(duration, ort_outs):
                # print(f"==>> 11111out_dur: {out_dur}")
                out_dur = out_dur[0]
                out_audio = out_audio[0] * np.array([volume], dtype=np.float32)
                if break_list:
                    center_sample = [0] * self.hop_length
                    for break_dur_index in break_list:
                        if text_list[break_dur_index] not in [1,7,8]:
                            raise ValueError(f'静音段索引错误，{break_list}，error index "{break_dur_index}"')
                        
                        center_index = int((sum(out_dur[:(break_dur_index + 1)]) + sum(out_dur[:break_dur_index])) // 2)
                        if out_dur[break_dur_index] == break_list[break_dur_index]:
                            continue
                        elif out_dur[break_dur_index] < break_list[break_dur_index]:
                            frame_num = int(break_list[break_dur_index] - out_dur[break_dur_index])
                            out_audio = np.concatenate((out_audio[:center_index * self.hop_length], np.tile(center_sample, frame_num), out_audio[center_index * self.hop_length:]))
                            out_dur[break_dur_index] += frame_num
                        else:
                            frame_num = int(out_dur[break_dur_index] - break_list[break_dur_index])
                            if frame_num >= out_dur[break_dur_index]:
                                continue
                            left_index = (center_index - (frame_num // 2)) * self.hop_length
                            right_index = (center_index + (frame_num // 2)) * self.hop_length
                            out_audio = np.concatenate((out_audio[:left_index], out_audio[right_index:]))
                            out_dur[break_dur_index] -= 2 * (frame_num // 2)
                #return out_audio, out_dur
                #print(out_dur)
                out_dur = [float(item) for item in out_dur]

                durs.extend(out_dur)
                # print(f"==>> out_dur: {out_dur}")

                out_audio_list.append(out_audio[:self.hop_length * int(sum(out_dur))])
        # return np.concatenate(out_audio_list), np.array(durs).astype(np.int32).tolist()
        return np.concatenate(out_audio_list), out_dur
    
    def convert_audio(self, audio_data, sample_rate=24000, bit_depth=16, channels=1, format='wav'):
        # 将NDArray转换为指定位深格式
        if bit_depth == 16:
            audio_data = np.asarray(audio_data * 32767, dtype=np.int16)
        elif bit_depth == 24:
            audio_data = np.asarray(audio_data * 8388607, dtype=np.int32)
        elif bit_depth == 32:
            audio_data = np.asarray(audio_data * 2147483647, dtype=np.int32)
        else:
            raise ValueError("Unsupported bit depth. Please use 16, 24, or 32.")
        
        audio_buffer = io.BytesIO()
        
        if format == 'pcm':
            audio_data = audio_data.tobytes()
            audio_buffer.write(audio_data)
            return audio_buffer.getvalue()
        elif format == 'wav':
            audio_segment = AudioSegment(
                audio_data.tobytes(), 
                frame_rate=sample_rate,
                sample_width=audio_data.dtype.itemsize, 
                channels=channels
            )
            audio_data = audio_segment.export(format='wav', bitrate='64k')
            audio_data = audio_data.read()
            audio_buffer.write(audio_data)
            return audio_buffer.getvalue()
        elif format == 'mp3':
            audio_segment = AudioSegment(
                audio_data.tobytes(), 
                frame_rate=sample_rate,
                sample_width=audio_data.dtype.itemsize, 
                channels=channels
            )
            audio_segment.export(audio_buffer, format='wav', bitrate='64k')
            audio_buffer.seek(0)
            process = subprocess.Popen(
                ['lame', '--silent', '-', '-'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            mp3_byte_stream, _ = process.communicate(input=audio_buffer.read())
            return mp3_byte_stream
        else:
            raise ValueError("Unsupported format. Please use 'wav', 'pcm', or 'mp3'.")

'''
# 示例用法
if __name__ == '__main__':
    # 读取 PCM 格式的音频文件
    with open('input.pcm', 'rb') as f:
        pcm_data = f.read()

    # 将 PCM 格式的音频数据转换为 MP3 格式
    try:
        converted_data = convert_audio(pcm_data, 'mp3')
        with open('output.mp3', 'wb') as f:
            f.write(converted_data)
        print("转换成功")
    except ValueError as e:
        print(e)
'''