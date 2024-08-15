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

    return phoneme_mask,tokens,polyphonics
    


def predict(onnx_session, ids,phoneme_mask, turnoff_tqdm=False):

       
    probs = onnx_session.run(
        [],
        {
            'input_ids': np.array(ids),
            'phoneme_mask': np.array(phoneme_mask)
        }
    )[0]

    preds = np.argmax(probs, axis=-1)

    return preds




providers = ['CUDAExecutionProvider']
sess_options = onnxruntime.SessionOptions()
sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
sess_options.intra_op_num_threads = 2
session_g2pw = onnxruntime.InferenceSession("/mnt/cfs/SPEECH/hupeng/git_loc_workspace/tal_frontend_service/tal_frontend/frontend/g2p/bertg2pw/g2p.onnx", sess_options=sess_options,providers=providers)

tokenizer = BertTokenizer.from_pretrained("/mnt/cfs/NLP/hub_models/bert-base-multilingual-cased")
polyphonic_chars_path = "/mnt/cfs/NLP/hsy/audio/project/bert/g2pW/saved_models/step/v2/POLYPHONIC_CHARS_0713.txt"
polyphonic_chars = [line.split('\t') for line in open(polyphonic_chars_path).read().strip().split('\n')]
labels, char2phonemes = get_phoneme_labels(polyphonic_chars)
length = len(labels)

sentences = ["it's       a good time"]
input_ids = []
phoneme_mask = []
batch_polyphonics = []
for sen in sentences:
    _phoneme_mask,tokens,polyphonics =  tokenize_and_map(tokenizer,sen,char2phonemes,length)
    tokens = ['[CLS]'] + tokens  + ['[SEP]']
    _phoneme_mask = [[0]*length] + _phoneme_mask + [[0]*length]
    input_ids.append(tokenizer.convert_tokens_to_ids(tokens))
    phoneme_mask.append(_phoneme_mask)
    batch_polyphonics.append(polyphonics)
    

preds = predict(session_g2pw, input_ids, phoneme_mask)
for spred,spoly in zip(preds,batch_polyphonics):
    spred = spred[1:-1]
    print(spoly)
    for pred,poly in zip(spred,spoly):
        if poly:
            print(labels[pred])
            
for spred, spoly, sen in zip(preds, batch_polyphonics, sentences):
    spred = spred[1:-1]
    sen = re.findall(r'[\u4e00-\u9fff]|[a-z]+[\'\-]?[a-z]+|[a-z ]', sen)
    last_lang = 'cn' if re.search(r'[\u4e00-\u9fff]', sen[0]) else 'en'
    print(spoly)
    for pred, poly, word in zip(spred, spoly, sen):
        print(word, pred, poly)