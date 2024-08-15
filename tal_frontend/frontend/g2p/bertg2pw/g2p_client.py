import re
import numpy as np
from transformers import BertTokenizer
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

from tal_frontend.frontend.g2p.bertg2pw.pinyin_dict import monophone
from tal_frontend.frontend.g2p.tone_sandhi import ToneSandhi
from tal_frontend.frontend.g2p.bertg2pw.tal_dict import tal_cn_dict
#from pinyin_dict import monophone

SM = set(
    ['b','p','m','f','d','t','n','l','g','k','h','j','q','x','zh','ch','sh','r','z','c','s','y','w']
    )

class TAL_G2P_Triton():
    def __init__(self,url):
        self.url = url
        self.model_name = "vits_g2p"
    
        self.tokenizer = BertTokenizer.from_pretrained("/mnt/cfs/NLP/hub_models/bert-base-multilingual-cased")
        
        self.polyphonic_chars_path = "/mnt/cfs/NLP/hsy/audio/project/bert/g2pW/saved_models/step/v2/POLYPHONIC_CHARS_0713.txt"
        self.polyphonic_chars = [line.split('\t') for line in open(self.polyphonic_chars_path).read().strip().split('\n')]
        self.labels, self.char2phonemes = self.get_phoneme_labels(self.polyphonic_chars)
        self.length = len(self.labels)
        self.poly_set = set()
        for i in self.polyphonic_chars:
            self.poly_set.add(i[0])
                    
        self.en_monophone = {}
        self.en_monophone_file = r'/mnt/cfs/SPEECH/hupeng/git_loc_workspace/tal_frontend_service/tal_frontend/frontend/g2p/bertg2pw/english_dict.list'
        with open(self.en_monophone_file, 'r', encoding='utf8') as fin:
            for line in fin:
                key, value = line.strip().split('\t', maxsplit=1)
                self.en_monophone[key] = value

        # 变调
        self.tonesandhi = ToneSandhi()
        
    def get_phoneme_labels(self, polyphonic_chars):
        labels = sorted(list(set([phoneme for char, phoneme in polyphonic_chars])))
        char2phonemes = {}
        for char, phoneme in polyphonic_chars:
            if char not in char2phonemes:
                char2phonemes[char] = []
            char2phonemes[char].append(labels.index(phoneme))
        return labels, char2phonemes
    
    def wordize_and_map(self, text:str):
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

        return phoneme_mask,tokens,polyphonics
    
    def syllable_split(self, syllable):
        tmp_sm = syllable[:2]
        if tmp_sm in SM:
            #return [tmp_sm, syllable[2:]]
            return f'{tmp_sm} {syllable[2:]}'
        elif syllable[0] in SM:
            #return [syllable[0], syllable[1:]]
            return f'{syllable[0]} {syllable[1:]}'
        else:
            return syllable
    
    def phoneme_convertor(self, phonemes:list):
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
    
    def infer(self, sentences):
        
        input_ids = []
        phoneme_mask = []
        batch_polyphonics = []
        
        for sen in sentences:
            sen = sen.replace('\'', '')
            _phoneme_mask,tokens,polyphonics =  self.tokenize_and_map(self.tokenizer, sen, self.char2phonemes, self.length)
            tokens = ['[CLS]'] + tokens  + ['[SEP]']
            _phoneme_mask = [[0]*self.length] + _phoneme_mask + [[0]*self.length]
            input_ids.append(self.tokenizer.convert_tokens_to_ids(tokens))
            phoneme_mask.append(_phoneme_mask)
            batch_polyphonics.append(polyphonics)
        
        input_ids = np.array(input_ids, dtype=np.int64)
        phoneme_mask = np.array(phoneme_mask, dtype=np.int64)
        
        inputs = [
        httpclient.InferInput("input_ids", input_ids.shape, "INT64"),
        httpclient.InferInput("phoneme_mask", phoneme_mask.shape, "INT64")]

        # Set input data
        inputs[0].set_data_from_numpy(input_ids)
        inputs[1].set_data_from_numpy(phoneme_mask)
        
    # Create output tensors to receive the results of inference
        outputs = [
        httpclient.InferRequestedOutput("probs") ]
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
        preds = np.argmax(probs, axis=-1)
        phonemes = [['', [], '']]
        for spred,spoly,sen in zip(preds,batch_polyphonics, sentences):
            spred = spred[1:-1]
            poly_phonemes = []
            for pred,poly in zip(spred,spoly):
                if poly:
                    poly_phonemes.append(self.labels[pred])

            poly_phonemes = self.phoneme_convertor(poly_phonemes)
            sen = sen.replace('\'', '')
            sen = re.findall(r'[\u4e00-\u9fff]|[a-z]+[\']?[a-z]+|[a-z]', sen)
            last_lang = 'cn' if re.search(r'[\u4e00-\u9fff]', sen[0]) else 'en'
            for word in sen:
                if re.match(r'[\u4e00-\u9fff]', word):
                    if monophone.get(ord(word)):
                        phoneme = monophone[ord(word)]
                        phoneme = self.syllable_split(phoneme)
                    elif word in self.poly_set:
                        phoneme = poly_phonemes.pop(0)
                    else:
                        phoneme = '#3'
                    lang = 'cn'
                else:
                    if self.en_monophone.get(word):
                        phoneme = self.en_monophone[word]
                    elif word in self.poly_set:
                        phoneme = poly_phonemes.pop(0)
                    else:
                        phoneme = '#3'
                    lang = 'en'
                    
                if lang == last_lang:
                    phonemes[-1][0] += f'{word} '
                    phonemes[-1][1].append(phoneme)
                    phonemes[-1][2] = lang
                else:
                    phonemes.append([f'{word} ', [phoneme], lang])
                    last_lang = lang
        res = []
        for seg in phonemes:
            content, phoneme, lang = seg
            if lang == 'cn':
                phoneme = phoneme
                phoneme = self.tonesandhi.modified_tone(content, phoneme)
            res.extend(phoneme)
        return res 
    
if __name__ == "__main__":

    url = "112.126.23.219:80"
    
    client = TAL_G2P_Triton(url)
    
    # Example input data (replace this with actual data)
    input_ids_batch = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64)  # Batch of 2 sequences
    phoneme_mask_batch = np.array([[[1], [1], [1]], [[0], [0], [0]]], dtype=np.int64)  # Corresponding mask
    
    sentencs = ['一行白鹭上青天hello world你好世界']
    client.infer(sentencs)

