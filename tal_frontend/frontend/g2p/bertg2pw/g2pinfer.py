import re
import onnxruntime
from transformers import BertTokenizer
import numpy as np

#from tal_frontend.frontend.g2p.bertg2pw.pinyin_dict import monophone
#from tal_frontend.frontend.g2p.utils import ph2id
#from tal_frontend.frontend.g2p.tone_sandhi import ToneSandhi
from pinyin_dict import monophone

SM = set(
    ['m','b','g','q','ch','z','l','sh','h','n','s','x','d','c','f','zh','k','r','t','p','j','y']
    )

class G2PW_INFER:
    def __init__(self) -> None:
        self.providers = ['CUDAExecutionProvider']
        self.sess_options = onnxruntime.SessionOptions()
        self.sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        self.sess_options.intra_op_num_threads = 2
        
        self.session_g2pw = onnxruntime.InferenceSession("/mnt/cfs/NLP/hsy/audio/project/bert/g2pW/saved_models/step/v2/g2p_0713.onnx", sess_options=self.sess_options,providers=self.providers)
        
        self.tokenizer = BertTokenizer.from_pretrained("/mnt/cfs/NLP/hub_models/bert-base-multilingual-cased")
        
        self.polyphonic_chars_path = "/mnt/cfs/NLP/hsy/audio/project/bert/g2pW/saved_models/step/v2/POLYPHONIC_CHARS_0713.txt"
        self.polyphonic_chars = [line.split('\t') for line in open(self.polyphonic_chars_path).read().strip().split('\n')]

        self.labels, self.char2phonemes = self.get_phoneme_labels(self.polyphonic_chars)
        self.length = len(self.labels)
        
        # 变调
        #self.tonesandhi = ToneSandhi()
        
        self.monoen_dict = {}
        self.monoen_file = r'/mnt/cfs/SPEECH/hupeng/git_loc_workspace/tal_frontend_service/tal_frontend/frontend/g2p/phonemes/en/en_monophone_tal.txt'
        with open(self.monoen_file, 'r', encoding='utf8') as fin:
            for line in fin:
                key, value = line.strip().split('\t', maxsplit=1)
                self.monoen_dict[key] = value
    
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
            
    def predict(self, sentences, turnoff_tqdm=False):
        print(sentences)
        input_ids = []
        phoneme_mask = []
        batch_polyphonics = []
        
        for sen in sentences:
            _phoneme_mask,tokens,polyphonics =  self.tokenize_and_map(self.tokenizer, sen, self.char2phonemes, self.length)
            tokens = ['[CLS]'] + tokens  + ['[SEP]']
            _phoneme_mask = [[0]*self.length] + _phoneme_mask + [[0]*self.length]
            input_ids.append(self.tokenizer.convert_tokens_to_ids(tokens))
            phoneme_mask.append(_phoneme_mask)
            batch_polyphonics.append(polyphonics)
            
        probs = self.session_g2pw.run(
            [],
            {
                'input_ids': np.array(input_ids),
                'phoneme_mask': np.array(phoneme_mask)
            }
        )[0]

        preds = np.argmax(probs, axis=-1)
        phonemes = [['', [], '']]
        print(sentences)
        
        for spred,spoly in zip(preds,batch_polyphonics):
            spred = spred[1:-1]
            for pred,poly in zip(spred,spoly):
                if poly:
                    phoneme = self.labels[pred]
                    
        for spred, spoly, sen in zip(preds, batch_polyphonics, sentences):
            spred = spred[1:-1]
            sen = re.findall(r'[\u4e00-\u9fff]|[a-z]+[\'\-]?[a-z]+|[a-z]', sen)
            last_lang = 'cn' if re.search(r'[\u4e00-\u9fff]', sen[0]) else 'en'
            for pred, poly in zip(spred, spoly):
                if re.match(r'[\u4e00-\u9fff]', word):
                    if poly:
                        phoneme = self.syllable_split(self.labels[pred])
                    else:
                        phoneme = monophone[ord(word)]
                        phoneme = self.syllable_split(phoneme)
                    lang = 'cn'
                else:
                    phoneme = self.monoen_dict[word]
                    lang = 'en'
                #phonemes.append(phoneme)
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
                #phoneme = self.tonesandhi.modified_tone(content, phoneme)
            res.extend(phoneme)     
        #phonemes_id = ph2id(phonemes)
        return res
            

if __name__ == '__main__':
    g2p = G2PW_INFER()
    sentences = ['一行白鹭上青天hello world你好世界']
    g2p.predict(sentences)
            
