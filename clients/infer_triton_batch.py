import numpy as np

from clients.tts_client import TAL_TTS_Triton
from tal_frontend.utils.errors import ValueLengthWarning

class TAL_TTS_Batch():
    def __init__(self, url, g2ppp_model_name, sample_rate=24000, hop_length=240):
        self.tal_tts_triton = TAL_TTS_Triton(url,g2ppp_model_name)
        self.sample_rate = sample_rate
        self.hop_length = hop_length

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

    def infer_batch(self, batch_data):
        
        target_length = 70
        """
        Perform batch inference on a list of data items.
        """
        # Initialize lists to collect inputs for each key
        text_list = []
        text_paddings = []
        text_padding_lengths = []
        sid_inputs = []
        dur_rates = []
        f0_rates = []
        rhy_list_paddings = []

        # Collect inputs for each data item in the batch
        for data in batch_data:
            # print(f"==>> data: {data}")
            spk_id, tts_type, _, text, dur_rate, f0_rate, _, _, rhy_list = data
            
            text_list.append(text)
            # Padding and collecting text
            text_padding = self.pad_list(text, target_length)
            text_paddings.append(text_padding)
            text_padding_lengths.append(len(text))
            
            # Collecting other inputs
            sid_inputs.append([spk_id])
            dur_rates.append([[1 / float(dur_rate)]])
            f0_rates.append([f0_rate])
            rhy_list_padding = self.pad_list(rhy_list, target_length=target_length)            
            rhy_list_paddings.append(rhy_list_padding)

        # Convert lists to numpy arrays and create input dictionary for batch
        input_dict = {
            "text": np.array(text_paddings).reshape(-1, target_length).astype(np.int32),  # (batch_size, 70
            
            "text_length":np.array(text_padding_lengths).reshape(-1, 1).astype(np.int32),
            "sid": np.array(sid_inputs).reshape(-1, 1).astype(np.int32),
            "dur_rate": np.array(dur_rates).astype(np.float32),
            "f0_rate": np.array(f0_rates).reshape(-1, 1).astype(np.float32),
            "rhy": np.array(rhy_list_paddings).reshape(-1, target_length).astype(np.int32)
        }

        # Perform batch inference
        spk_id = batch_data[0][0]
        #tts_type = self.type_map[str(spk_id)]
        ort_outs, duration, f0 = self.tal_tts_triton.get_result(input_dict, tts_type=tts_type)
      
        
        # out_dur = duration[0][0]
        # # 去除最后所有为0的元素
        # non_zero_index = min(len(text), target_length)  
        # # 截取非零部分
        # out_dur = out_dur[:non_zero_index]
        # duration = out_dur.reshape(1, 1, len(out_dur))
        

        # Initialize lists to store results
        processed_durations = []
        # Loop through each item in the batch
        for i in range(len(batch_data)):
            out_dur = duration[i][0]
            # 去除最后所有为0的元素
            non_zero_index = min(len(text_list[i]), target_length)  
            
            # 截取非零部分
            out_dur = out_dur[:non_zero_index]
            
            # Reshape and store the result
            processed_duration = out_dur.reshape(1, 1, len(out_dur))
            processed_durations.append(processed_duration)

        # Convert list of processed durations to numpy array or any required format
        # processed_durations = np.array(processed_durations).reshape(-1, 1, len(out_dur))
        # print(f"==>> (ort_outs, processed_durations: {ort_outs.shape, processed_durations.shape}")
        ort_outs = np.expand_dims(ort_outs, axis=1)
        return (ort_outs, processed_durations)

    def run_inference(self, datas):
        """
        Run batch inference on a list of data.
        """
        # Split the data into chunks if necessary to fit GPU memory or other constraints
        # batch_size = 8  # Define your preferred batch size
        all_audio = []

        # for i in range(0, len(datas), batch_size):
            # batch = datas[i:i+batch_size]
        ort_outs_batch, durations_batch = self.infer_batch(datas)
        #raise ValueLengthWarning
        # Process each output from the batch and append to all_audio
        for ort_outs, duration in zip(ort_outs_batch, durations_batch):
            duration_frame = np.sum(duration)
            # print('==>>Duration: ', duration)
            # print(duration_frame)
            if duration_frame > 1000:
                raise ValueLengthWarning(f'合成文本过长')
            # Assuming you need to further process ort_outs and duration before appending
            # Here we just append them directly for simplicity
            all_audio.append((ort_outs, duration))

        return all_audio


if __name__ == "__main__":
    url = '47.94.0.184:8000'

    tal_tts = TAL_TTS_Batch(url, sample_rate=24000, hop_length=240,  output_dir='')

    datas = [['1', 'tal_vits', 1, [1, 21, 213, 19, 275, 18, 155, 12, 275, 8, 8], '1.0', '1.0', '100', {}, [1, 1, 2, 1, 3, 1, 2, 1, 6, 1, 1]]]


    # infer_result = tal_tts.infer(text)
    infer_result = tal_tts.run_inference(datas)
    print('finished')
    # del infer_result['ort_outs']
    # print(f"==>> time_log: {infer_result}")
