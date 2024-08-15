import re
# split the text to sentence.
# Created by HuangLiu on 2024.06.05.
SENTENCE_SPLITOR = re.compile(r'([：、，；。？！,;?!:][”’]?)')
SENTENCE_END_PUNCT_SPLITOR = re.compile(r'([；。？！;?!])')
punct = "：、，；。？！,;?!:"
end_punct = "；。？！;?!"
puncts = [',', '，', ':', '：', '。', '.', '!', '！', '?', '？', ';','；','、']

def is_number(text_string):
  """判断一个传入字符是否是数字"""
  if re.search(r'(^#|^\d)', text_string):
  #if re.search(r'^\d', text_string):
    return True
  else:
	  return False

def is_mandarin(uchar):
  """判断一个unicode是否是汉字"""
  code_point = ord(uchar)
  if code_point >= 0x4e00 and code_point <= 0x9fff:
    return True
  else:
    return False

def is_english_for_spss(text_string):
  """判断一个传入字符是否是英文"""
  if re.search(r'^[a-zA-Z]', text_string):
    return True
  else:
	  return False

def is_punct_for_spss(text_string):
	"""判断一个传入字符是否是spss接受的标点，传入可能是字符串"""
	if len(text_string) > 1:
		return False
	if text_string in puncts:
		return True
	else:
		return False
       
def remove_illegal_punct(text):
  '''remove illegal punct: [].='''
  new_text_seq = ''
  for t in text:
    if t in [' ','	']:
      new_text_seq += t
    if is_number(t):
      new_text_seq += t
    if is_mandarin(t):
      new_text_seq += t
    if is_english_for_spss(t):
      new_text_seq += t
    if is_punct_for_spss(t):
      new_text_seq += t
  return new_text_seq
  
def prosody_text_list2sequence(prosdy_text_list):
  '''for text list, we convert to sequence
  ['你', '#2'] -> 你#2
  '''
  prosody_text_sequence = ""
  for item in prosdy_text_list:
    prosody_text_sequence += ' ' + item + ' '
  prosody_text_sequence = re.sub(r'\s+',r' ', prosody_text_sequence).strip()

  # 你 好
  prosody_text_sequence = re.sub(r'([^a-zA-Z])\s+', r'\1', prosody_text_sequence) 
  prosody_text_sequence = re.sub(r'\s+([^a-zA-Z])', r'\1', prosody_text_sequence)
  return prosody_text_sequence

def text_process(text_seq):
  '''text process to list'''
  # for text pinyin
  text_list = re.sub(r'(#\d)', r' \1 ', text_seq)   
  # text_list = re.sub(r'([a-zAZ]+)(\')(s)', r'\1\3', text_list)  # remove the \' from \'s.
  # Characters that are not in the range of the set can be matched by negation, ^ represent negation.
  # \u400e-\u9fa5 : Chinese characters range
  # text_list = re.sub(r'[^a-zA-Z\u400e-\u9fa5 ]', ' ', text_list)  
  # for not chinese and english characters

  text_list = re.sub(r'([^a-zA-Z\u400e-\u9fa5 \'])', r' \1 ', text_list) 
  text_list = re.sub(r'\'', r'', text_list)

  # add the blanks for words (whether is chinese characters)
  text_list = re.sub(r'([\u400e-\u9fa5])', r' \1 ', text_list)
  text_list = re.sub(r'\s+', ' ', text_list).strip()
  text_list = re.sub(r'(#)\s(\d)', r'\1\2', text_list)   # combine the split rhythm.
  text_list = text_list.split()
  text_list = [text for text in text_list if text != '']
  return text_list

def list_has_word(text_list):
  for item in text_list:
    if re.search(r'[a-zA-Z\u400e-\u9fa5\d]', item):
      return True
  return False

class TextSplit:
  """
  split text to sentence. 
  """
  def __init__(self, max_puretext_len=10, max_sent_len=12, max_pure_entext_len=35):
    '''find the segment pair times from the ctm file
      Args:
        max_punct_len: for continus pure_text, use max_len to restrict it's length.
        max_sent_len: sent's max text length.
        max_pure_entext_len: for continus pure_en_text, use max_pure_entext_len to restrict it's length.
      Returns:
        the split sentence
    '''
    self.max_puretext_len = max_puretext_len
    self.max_sent_len = max_sent_len
    self.max_pure_entext_len = max_pure_entext_len
  
  def add_punct2pure_text(self, text) -> str:
    "add punct to continus text"
    new_text = []
    pure_text = 0 # count continue pure text
    pure_text_en = 0 # count continue pure en text
    text_list = text_process(text)
    len_text = len(text_list)
    for i in range(len_text):
      item = text_list[i]
      new_text += [item]
      if re.search(r'#\d', item):
        continue
      elif item not in punct: # origin is punct, we changed
        pure_text += 1
        if is_english_for_spss(item):
          pure_text_en += 1
        else:
          pure_text_en = 0
      else:
        pure_text = 0
        pure_text_en = 0
      if pure_text >= self.max_puretext_len or pure_text_en >= self.max_pure_entext_len:
        # item是单个字符时：if current item belongs to a english word, we not add punct.
        # if re.search(r'[a-zA-Z]', item) and i+1 < len_text and re.search(r'[a-zA-Z]', text_list[i+1]):
        #   pass
        # else:
        pure_text = 0
        pure_text_en = 0
        new_text += ['。']
    new_text = prosody_text_list2sequence(new_text)
    return new_text
  
  def split_text2sentence(self,text) -> list:
    "split by punct"
    text = SENTENCE_SPLITOR.sub(r'\1\n', text) # origin use SENTENCE_SPLITOR
    text = text.strip()
    sentences = [sentence.strip() for sentence in re.split(r'\n+', text)]
    return sentences

  def merge_sentence(self, sentence_list) -> list:
    "merge samll length sentence"
    new_sentence_list = []
    n = len(sentence_list)
    i = 0
    
    acc_len = 0
    tmp_sent = []
    while i < n:
      text_list = text_process(sentence_list[i])
      cur_len = len(text_list)
      # count pure_text_en num
      # 加入英文单词个数超过一定数量，该句不合并,直接单独作为一句或者并入之前的句子。
      pure_text_en = 0
      for item in text_list:
        if is_english_for_spss(item):
          pure_text_en += 1
      
      # 加入英文单词个数超过一定数量，该句不合并,直接单独作为一句或者并入之前的句子。
      # 英文断句优先级要更前面
      if pure_text_en >= self.max_pure_entext_len:
        tmp_sent += text_list
        new_sentence_list.append(prosody_text_list2sequence(tmp_sent))
        acc_len = 0
        tmp_sent = []
      elif acc_len + cur_len < self.max_sent_len:
        acc_len += cur_len
        if list_has_word(text_list):
          tmp_sent += text_list
      else:
        if list_has_word(tmp_sent):
          new_sentence_list.append(prosody_text_list2sequence(tmp_sent))
        tmp_sent = text_list
        acc_len = cur_len
      i += 1
    new_sentence_list.append(prosody_text_list2sequence(tmp_sent))
    return new_sentence_list

  def split_en(self, sentence_list) -> list:
    "split the english dominates sentences"
    new_sentence_list = []
    n = len(sentence_list)
    i = 0
    
    acc_len = 0
    tmp_sent = []
    while i < n:
      current_sentence_list = text_process(sentence_list[i])
      cur_len = len(current_sentence_list)
      # count pure_text_en num
      pure_text_en = 0
      for item in current_sentence_list:
        if is_english_for_spss(item):
          pure_text_en += 1
      
      # 加入英文单词个数超过一定数量，直接用标点符号分割句子。
      if pure_text_en >= self.max_pure_entext_len:
        current_text = sentence_list[i]
        current_text = re.sub(r'([，。？：；、！,?:;!.])', r'\1|', current_text)
        # 对于特殊的Mr. Mrs.等，不分割，前面先加'|'后判断去除'|'
        current_text = re.sub(r'(Mr.)\|', r'\1', current_text)
        current_text = re.sub(r'(Mrs.)\|', r'\1', current_text)
        en_list = current_text.split('|')
        for item in en_list:
          has_chzh_characters_flag = re.search(r'[a-zA-Z\u400e-\u9fa5\d]', item)
          if has_chzh_characters_flag:
            new_sentence_list.append(item)
      else:
        if list_has_word(current_sentence_list):
          new_sentence_list.append(sentence_list[i])
      i += 1

    return new_sentence_list
  
  def split(self,text):
    text = text.strip()
    text = re.sub(r'(\n)+', r'\n', text)
    # if not remove_illegal_punct, egs: article “Machine -> article“Machine
    text = remove_illegal_punct(text)
    text = self.add_punct2pure_text(text)
    # 标点符号分割：添加'|'作为分隔符，为了保留标点符号
    text = re.sub(r'([，。？：；、！,?:;!.])', r'\1|', text)
    # 对于特殊的Mr. Mrs.等，不分割，前面先加'|'后判断去除'|'
    text = re.sub(r'(Mr.)\|', r'\1', text)
    text = re.sub(r'(Mrs.)\|', r'\1', text)

    # 切分句子
    sentence_list = text.split('|')
    # 对标点符号内的句子进一步判断是否超过了
    # sentence_list = self.split_text2sentence(text)

    ### important, this will merge the small length sentence.
    sentence_list = self.merge_sentence(sentence_list)
    # sentence_list = self.split_en(sentence_list)
    return sentence_list
  
  def split_punct(self,text):
    text = text.strip()
    sentence_list = re.split(r'([，。？：；、！,?:;!])', text)
    if len(sentence_list) > 1:
      sentence_list = ["".join(i) for i in zip(sentence_list[0::2],sentence_list[1::2])]
    sentence_list = [item for item in sentence_list if item !=""]
    return sentence_list


if __name__ == '__main__':
    ts = TextSplit()
    text = "你好小思我是好未来的员工你有什么问题想要问我吗我可以帮你解决所有问题比如三行四列行列式怎么解比如一行白鹭上青天是什么意思比如李白的好朋友有哪些你喜欢吃什么呀小思红烧豆腐怎么做我是一个乖宝宝你不是嘻嘻嘻嘻嘻华为手机天下无敌用华为不卡"
    res = ts.split(text)
    for i in res:
        print(len(i))
    print(res)
    