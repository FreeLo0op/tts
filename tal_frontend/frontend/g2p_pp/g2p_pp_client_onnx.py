
#先安装客户端
#pip install tritonclient\[all\]
import re
import numpy as np
from transformers import BertTokenizer
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from tal_frontend.frontend.g2p.bertg2pw.pinyin_dict import cn_monophone
from tal_frontend.frontend.g2p.tone_sandhi import ToneSandhi
from tal_frontend.frontend.g2p.bertg2pw.tal_dict import tal_cn_dict

class TAL_G2PPP_Triton():
    def __init__(self,url):
        self.url = url
        self.model_name = "vits_rhy"
        self.labels_rhy = ["","#0","#1","#2","#3"]
        # self.tokenizer = BertTokenizer.from_pretrained("/mnt/cfs/NLP/hub_models/bert-base-multilingual-cased")
        # self.tokenizer = BertTokenizer.from_pretrained("/models/asr_models/bert_model/bert-base-multilingual-cased")
        self.tokenizer = BertTokenizer.from_pretrained("/mnt/cfs/NLP/hub_models/bert-base-multilingual-cased")
        
        self.polyphonic_chars_path = "tal_frontend/frontend/g2p_pp/POLYPHONIC_CHARS.txt"
        self.polyphonic_chars = [line.split('\t') for line in open(self.polyphonic_chars_path).read().strip().split('\n')]
        self.labels_pho, self.char2phonemes = self.get_phoneme_labels(self.polyphonic_chars)
        self.length = len(self.labels_pho)
        
        self.en_monophone = {}
        self.en_monophone_file = r'tal_frontend/frontend/g2p/bertg2pw/english_dict.list'
        with open(self.en_monophone_file, 'r', encoding='utf8') as fin:
            for line in fin:
                key, value = line.strip().split('\t', maxsplit=1)
                self.en_monophone[key] = value
                
        self.tonesandhi = ToneSandhi()
        
        self.sm = set(
                        ['b','p','m','f','d','t',
                        'n','l','g','k','h','j',
                        'q','x','zh','ch','sh',
                        'r','z','c','s','y','w']
                    )

    def syllable_split(self, syllable):
        tmp_sm = syllable[:2]
        if tmp_sm in self.sm:
            #return [tmp_sm, syllable[2:]]
            return f'{tmp_sm} {syllable[2:]}'
        elif syllable[0] in self.sm:
            #return [syllable[0], syllable[1:]]
            return f'{syllable[0]} {syllable[1:]}'
        else:
            return syllable
    
    def phoneme_convertor(self, phoneme):
        if phoneme.isupper():  
            return phoneme
        else:
            if phoneme == 'xer':
                return 'er2'
            else:
                tone = phoneme[-1]
                phoneme = phoneme[:-1]
                phoneme = tal_cn_dict[phoneme] + tone
                return phoneme
        '''
        tal_phonemes = []
        for phoneme in phonemes:
            if phoneme.isupper():    
                tal_phonemes.append(phoneme)
            else:
                if phoneme == 'xer':
                    tal_phonemes.append('er2')
                else:
                    tone = phoneme[-1]
                    phoneme = phoneme[:-1]
                    phoneme = tal_cn_dict[phoneme] + tone
                    tal_phonemes.append(phoneme)
        return tal_phonemes
        '''
        
    def get_phoneme_labels(self, polyphonic_chars):
        labels = sorted(list(set([phoneme for char, phoneme in polyphonic_chars])))
        char2phonemes = {}
        for char, phoneme in polyphonic_chars:
            if char not in char2phonemes:
                char2phonemes[char] = []
            char2phonemes[char].append(labels.index(phoneme))
        return labels, char2phonemes
    
    def wordize_and_map(self, text):
        words = []
        index_map_from_text_to_word = []
        index_map_from_word_to_text = []
        while len(text) > 0:
            match_space = re.match(r'^ +', text)
            if match_space:
                space_str = match_space.group(0)
                index_map_from_text_to_word += [None] * len(space_str)
                text = text[len(space_str):]
                continue

            match_en = re.match(r'^[a-zA-Z0-9]+', text)
            if match_en:
                en_word = match_en.group(0)

                word_start_pos = len(index_map_from_text_to_word)
                word_end_pos = word_start_pos + len(en_word)
                index_map_from_word_to_text.append((word_start_pos, word_end_pos))

                index_map_from_text_to_word += [len(words)] * len(en_word)

                words.append(en_word)
                text = text[len(en_word):]
            else:
                word_start_pos = len(index_map_from_text_to_word)
                word_end_pos = word_start_pos + 1
                index_map_from_word_to_text.append((word_start_pos, word_end_pos))

                index_map_from_text_to_word += [len(words)]

                words.append(text[0])
                text = text[1:]
        return words, index_map_from_text_to_word, index_map_from_word_to_text

    def tokenize_and_map(self, tokenizer, text, maps = {},length=0):
        words, text2word, word2text = self.wordize_and_map(text)

        tokens = []
        index_map_from_token_to_text = []
        phoneme_mask = []
        polyphonics = []
        for word, (word_start, word_end) in zip(words, word2text):
            word_tokens = tokenizer.tokenize(word)
            pre_phoneme_mask = [0] * length
            if len(word_tokens) == 0 or word_tokens == ['[UNK]']:
                index_map_from_token_to_text.append((word_start, word_end))
                tokens.append('[UNK]')
                polyphonics.append(False)
                phoneme_mask.append(pre_phoneme_mask)
            else:
                current_word_start = word_start
                for word_token in word_tokens:
                    word_token_len = len(re.sub(r'^##', '', word_token))
                    index_map_from_token_to_text.append(
                        (current_word_start, current_word_start + word_token_len))
                    current_word_start = current_word_start + word_token_len
                    tokens.append(word_token)
                for _ in word_tokens[1:]:
                    phoneme_mask.append(pre_phoneme_mask)
                    polyphonics.append(False)
                
                if word in maps:
                    phoneme_mask.append([1 if i in maps[word] else 0 for i in range(length)])
                    polyphonics.append(True)
                else:
                    phoneme_mask.append([0] * length)
                    polyphonics.append(False)
        index_map_from_text_to_token = text2word
        for i, (token_start, token_end) in enumerate(index_map_from_token_to_text):
            for token_pos in range(token_start, token_end):
                index_map_from_text_to_token[token_pos] = i

        return phoneme_mask,tokens,polyphonics,word2text,index_map_from_text_to_token

    def pp_post_process(self, rhys:list, phonemes:list):
        #[' i4', 'x ing2', 'b ai2', 'l u4', 'sh ang4', 'q ing1', 't ian1', 'sp3', 'HH AH0 L OW1', 'WER1 L D', 'n i2', 'h ao3', 'Z IY1', 'sh iii4', 'j ie4']
        #['#0', '#1', '#0', '#0', '#1', '#0', '#3', '0', '#1', '#3', '#0', '#1', '#1', '#0', '#0']
        rhy_res = []
        phonemes_res = []
        for rhy, phoneme in zip(rhys, phonemes):
            phoneme = phoneme.split()
            tmp_rhy = ['0'] * (len(phoneme))
            tmp_rhy[-1] = rhy
            rhy_res.extend(tmp_rhy)
            phonemes_res.extend(phoneme)
        
        return rhy_res, phonemes_res
    
    def pad_array(self, arr, max_len=70):
        max_len = min(max(len(sub_arr) for sub_arr in arr),max_len)
        padded_arrays = []
        for sub_arr in arr:
            sub_arr = sub_arr[:max_len]
            if isinstance(sub_arr[0], list):  # 检查是否是二维子数组
                # 对二维子数组进行填充
                padded_sub_arr = np.pad(sub_arr, ((0, max_len - len(sub_arr)), (0, 0)), mode='constant', constant_values=0)
            else:  # 一维子数组
                # 对一维子数组进行填充
                padded_sub_arr = np.pad(sub_arr, (0, max_len - len(sub_arr)), mode='constant', constant_values=0)
            padded_arrays.append(padded_sub_arr)
        return np.array(padded_arrays)
    
    def infer(self, sentences):
        
        input_ids = []
        phoneme_mask = []
        word2texts = []
        text2tokens = []
        batch_polyphonics = []
        
        for sen in sentences:
            _phoneme_mask,tokens,polyphonics,word2text,text2token = self.tokenize_and_map(self.tokenizer, sen, self.char2phonemes, self.length)
            tokens = ['[CLS]'] + tokens  + ['[SEP]']
            _phoneme_mask = [[0]* self.length] + _phoneme_mask + [[0]* self.length]
            input_ids.append(self.tokenizer.convert_tokens_to_ids(tokens))
            phoneme_mask.append(_phoneme_mask)
            word2texts.append(word2text)
            text2tokens.append(text2token)
            batch_polyphonics.append(polyphonics)
        
        #input_ids = np.array(input_ids, dtype=np.int64)
        #phoneme_mask = np.array(phoneme_mask, dtype=np.int64)
        attention_mask = self.pad_array([[1]*len(_id) for _id in input_ids])
        input_ids = self.pad_array(input_ids)
        phoneme_mask = self.pad_array(phoneme_mask)
        
        inputs = [
                httpclient.InferInput("input_ids", input_ids.shape, "INT64"),
                httpclient.InferInput("attention_mask", attention_mask.shape, "INT64"),
                httpclient.InferInput("phoneme_mask", phoneme_mask.shape, "INT64")
        ]

        # Set input data
        inputs[0].set_data_from_numpy(input_ids)
        inputs[1].set_data_from_numpy(attention_mask)
        inputs[2].set_data_from_numpy(phoneme_mask)
    # Create output tensors to receive the results of inference
    
# outputs.append(httpclient.InferRequestedOutput('probs_rhy'))
# outputs.append(httpclient.InferRequestedOutput('probs_pho'))
        outputs = [
            httpclient.InferRequestedOutput("probs_rhy"),
            httpclient.InferRequestedOutput("probs_pho"),
        
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
        probs_rhy = results.as_numpy("probs_rhy")
        ## print(f"==>> probs_rhy: {probs_rhy.shape}")
        probs_pho = results.as_numpy("probs_pho")
        ## print(f"==>> probs_pho: {probs_pho.shape}")
        preds_pho = np.argmax(probs_pho, axis=-1)
        preds_rhy = np.argmax(probs_rhy, axis=-1)
        
        for sen,pred_pho,pred_rhy,word2text,text2token,is_polyphonics in zip(sentences,preds_pho,preds_rhy,word2texts,text2tokens,batch_polyphonics):
            pred_pho = pred_pho[1:-1]
            pred_rhy = pred_rhy[1:-1]
            last_index = 0
            res = []
            for i,(start,end) in enumerate(word2text):
                if start > last_index:
                    pre_text = sen[last_index:start]
                    res.append({pre_text:{"pho":"","rhy":0}})
                current_text = sen[start:end]
                current_token_index = text2token[end-1]
                if is_polyphonics[current_token_index]:
                    current_pho = self.labels_pho[pred_pho[current_token_index]]
                else:
                    current_pho = ""
                current_rhy = self.labels_rhy[pred_rhy[current_token_index]]
                last_index = end
                res.append({current_text:{"pho":current_pho,"rhy":current_rhy}})

        
        phonemes = [['', [], '']]
        prosodies = []
        last_lang = 'cn' if re.search(r'[\u4e00-\u9fff]', list(res[0])[0]) else 'en'
        for item in res:
            for word, value in item.items():
                pho, rhy = value['pho'], value['rhy']
                if re.match(r'[ ]+', word):
                    continue
                elif re.match(r'[\,\.\?\:\!，。；：、！？]', word):
                    prosodies.append('0')
                    pho = 'sp3'
                    lang = 'en'
                elif re.match(r'[\u4e00-\u9fff]', word):
                    if pho:
                        pho = self.phoneme_convertor(pho)
                    elif cn_monophone.get(ord(word)):
                        pho = cn_monophone[ord(word)]
                        pho = self.syllable_split(pho)
                    else:
                        pho = '#3'
                    lang = 'cn'
                    prosodies.append(rhy)
                else:
                    if pho:
                        pass
                    elif self.en_monophone.get(word):
                        pho = self.en_monophone[word]
                    else:
                        pho = '#3'
                    lang = 'en'
                    prosodies.append(rhy)
                if lang == last_lang:
                    phonemes[-1][0] += f'{word} '
                    phonemes[-1][1].append(pho)
                    phonemes[-1][2] = lang
                else:
                    phonemes.append([f'{word} ', [pho], lang])
                    last_lang = lang

        modified_phonemes = []
        for seg in phonemes:
            content, phoneme, lang = seg
            if lang == 'cn':
                phoneme = self.tonesandhi.modified_tone(content, phoneme)
            modified_phonemes.extend(phoneme)

        
        rhy_res, phonemes_res = self.pp_post_process(prosodies, modified_phonemes)

        assert len(rhy_res) == len(phonemes_res)
        return rhy_res, phonemes_res
    
if __name__ == "__main__":

    url = "123.56.235.205:80"
    
    client = TAL_G2PPP_Triton(url)
    
    sentences = ["一行白鹭上青天，hello world你好z世界"]
    
    probs_batch = client.infer(sentences)
    '''
    try:
        probs_batch = client.infer(sentences)
        print("Inference result:", probs_batch)
    except Exception as e:
        print(f"An error occurred during inference: {e}")
    '''
    '''
    [{'一': {'pho': 'yi4', 'rhy': '#0'}}, {'行': {'pho': 'xing2', 'rhy': '#1'}}, {'白': {'pho': 'bai2', 'rhy': '#0'}}, {'鹭': {'pho': '', 'rhy': '#0'}}, {'上': {'pho': 'shang4', 'rhy': '#1'}}, {'青': {'pho': 'qing1', 'rhy': '#0'}}, {'天': {'pho': '', 'rhy': '#3'}}, {'，': {'pho': '', 'rhy': ''}}, {'hello': {'pho': 'HH AH0 L OW1', 'rhy': '#1'}}, {' ': {'pho': '', 'rhy': 0}}, {'world': {'pho': '', 'rhy': '#3'}}, {'你': {'pho': '', 'rhy': '#0'}}, {'好': {'pho': 'hao3', 'rhy': '#1'}}, {'z': {'pho': '', 'rhy': '#1'}}, {'世': {'pho': '', 'rhy': '#0'}}, {' 界': {'pho': '', 'rhy': '#0'}}]
    '''
