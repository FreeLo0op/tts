import onnxruntime
from transformers import BertTokenizer
from tqdm import tqdm
import numpy as np  
import re

def get_phoneme_labels(polyphonic_chars):
    labels = sorted(list(set([phoneme for char, phoneme in polyphonic_chars])))
    char2phonemes = {}
    for char, phoneme in polyphonic_chars:
        if char not in char2phonemes:
            char2phonemes[char] = []
        char2phonemes[char].append(labels.index(phoneme))
    return labels, char2phonemes

def wordize_and_map(text):
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


def tokenize_and_map(tokenizer, text, maps = {},length=0):
    words, text2word, word2text = wordize_and_map(text)

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
    


def predict(onnx_session, ids,phoneme_mask, turnoff_tqdm=False):

       
    probs_pho,probs_rhy = onnx_session.run(
        [],
        {
            'input_ids': np.array(ids),
            'phoneme_mask': np.array(phoneme_mask)
        }
    )

    preds_pho = np.argmax(probs_pho, axis=-1)
    preds_rhy = np.argmax(probs_rhy, axis=-1)
    return preds_pho,preds_rhy




providers = ['CUDAExecutionProvider']
sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
sess_options.intra_op_num_threads = 2
session_g2pw = onnxruntime.InferenceSession("/mnt/cfs/NLP/hsy/audio/project/bert/joint/saved_models/onnx/joint.onnx", sess_options=sess_options,providers=providers)


tokenizer = BertTokenizer.from_pretrained("/mnt/cfs/NLP/hub_models/bert-base-multilingual-cased")
polyphonic_chars_path = "/mnt/cfs/NLP/hsy/audio/project/bert/joint/data_process/POLYPHONIC_CHARS.txt"
polyphonic_chars = [line.split('\t') for line in open(polyphonic_chars_path).read().strip().split('\n')]
labels_pho, char2phonemes = get_phoneme_labels(polyphonic_chars)
labels_rhy = ["","#0","#1","#2","#3"]
length = len(labels_pho)

sentences = ["一行白鹭上青天hello world你好z世界"]
input_ids = []
phoneme_mask = []
word2texts = []
text2tokens = []
batch_polyphonics = []
for sen in sentences:
    _phoneme_mask,tokens,polyphonics,word2text,text2token = tokenize_and_map(tokenizer,sen,char2phonemes,length)
    tokens = ['[CLS]'] + tokens  + ['[SEP]']
    _phoneme_mask = [[0]*length] + _phoneme_mask + [[0]*length]
    input_ids.append(tokenizer.convert_tokens_to_ids(tokens))
    phoneme_mask.append(_phoneme_mask)
    word2texts.append(word2text)
    text2tokens.append(text2token)
    batch_polyphonics.append(polyphonics)

preds_pho,preds_rhy = predict(session_g2pw, input_ids, phoneme_mask)

# preds_pho -> (batch_size,len(sentence),n) 
# pred_rhy -> (batch_size,len(sentence),5) 
# word2text -> (batch_size,(start_index,end_index))
# text2token -> (batch_size,len(sentence))


            
for sen,pred_pho,pred_rhy,word2text,text2token,is_polyphonics in zip(sentences,preds_pho,preds_rhy,word2texts,text2tokens,batch_polyphonics):
    pred_pho = pred_pho[1:-1]
    pred_rhy = pred_rhy[1:-1]
    print(len(pred_pho),len(pred_rhy),len(word2text),len(text2token),len(is_polyphonics))
    last_index = 0
    res = []
    for i,(start,end) in enumerate(word2text):
        if start > last_index:
            pre_text = sen[last_index:start]
            res.append({pre_text:{"pho":"","rhy":0}})
        current_text = sen[start:end]
        current_token_index = text2token[end-1]
        if is_polyphonics[current_token_index]:
            current_pho = labels_pho[pred_pho[current_token_index]]
        else:
            current_pho = ""
        current_rhy = labels_rhy[pred_rhy[current_token_index]]
        last_index = end
        res.append({current_text:{"pho":current_pho,"rhy":current_rhy}})
    print(res)