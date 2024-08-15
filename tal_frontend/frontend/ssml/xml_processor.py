# -*- coding: utf-8 -*-
import re
import xml.dom.minidom
import xml.parsers.expat
from xml.dom.minidom import Node
from xml.dom.minidom import parseString
from tal_frontend.frontend.ssml.tal_dict import tal_cn_dict

class MixTextProcessor():
    def __repr__(self):
        print("@an MixTextProcessor class")

    def get_xml_content(self, mixstr):
        '''返回字符串的 xml 内容'''
        xmlptn = re.compile(r"<speak>.*?</speak>", re.M | re.S)
        ctn = re.search(xmlptn, mixstr)
        if ctn:
            return ctn.group(0)
        else:
            return None

    def get_content_split(self, mixstr):
        ''' 文本分解，顺序加了列表中，按非 xml 和 xml 分开，对应的字符串,带标点符号
        不能去除空格，因为 xml 中tag 属性带空格
        '''
        ctlist = []
        # print("Testing:",mixstr[:20])
        patn = re.compile(r'(.*\s*?)(<speak>.*?</speak>)(.*\s*)$', re.M | re.S)
        mat = re.match(patn, mixstr)
        if mat:
            pre_xml = mat.group(1)
            in_xml = mat.group(2)
            after_xml = mat.group(3)

            ctlist.append(pre_xml)
            ctlist.append(in_xml)
            ctlist.append(after_xml)
            return ctlist
        else:
            ctlist.append(mixstr)
        return ctlist

    @classmethod
    def get_pinyin_split(self, mixstr):
        ctlist = []
        patn = re.compile(r'(.*\s*?)(<speak>.*?</speak>)(.*\s*)$', re.M | re.S)
        mat = re.match(patn, mixstr)
        if mat:
            # pre <speak>
            pre_xml = mat.group(1)
            # between <speak> ... </speak>
            in_xml = mat.group(2)
            # post </speak>
            after_xml = mat.group(3)

            # pre with none syllable
            if pre_xml:
                ctlist.append([pre_xml, []])

            # between with syllable
            # [(sub sentence, [syllables]), ...]
            dom = DomXml(in_xml)
            pinyinlist = dom.get_pinyins_for_xml()
            ctlist = ctlist + pinyinlist

            # post with none syllable
            if after_xml:
                ctlist.append([after_xml, []])
        else:
            ctlist.append([mixstr, []])

        return ctlist

    @classmethod
    def get_dom_split(self, mixstr):
        ''' 文本分解，顺序加了列表中，返回文本和say-as标签
        '''
        ctlist = []
        patn = re.compile(r'(.*\s*?)(<speak>.*?</speak>)(.*\s*)$', re.M | re.S)
        mat = re.match(patn, mixstr)
        if mat:
            pre_xml = mat.group(1)
            in_xml = mat.group(2)
            after_xml = mat.group(3)

            if pre_xml:
                ctlist.append(pre_xml)

            dom = DomXml(in_xml)
            tags = dom.get_text_and_sayas_tags()
            ctlist.extend(tags)

            if after_xml:
                ctlist.append(after_xml)
        else:
            ctlist.append(mixstr)

        return ctlist

class DomXml():
    def __init__(self, xmlstr):
        self.tdom = parseString(xmlstr)  #Document
        self.root = self.tdom.documentElement  #Element
        self.rnode = self.tdom.childNodes  #NodeList
        self.emotions = {'中性': 'neutral', '开心': 'happy', '生气': 'angry', '疑问': 'confused', '难过': 'sad', '害怕': 'fear'}
        self.valid_emotions = set(self.emotions.values())
        self.pinyin_pattern = r'([a-z]+)(\d+)'
        
    def get_speak_info(self) -> dict :
        attributes = {}
        if self.root.hasAttributes():
            for attr_name, attr_value in self.root.attributes.items():
                attributes[attr_name] = attr_value
                
        emotion = attributes.get('emotion', 'neutral')
        #if emotion not in self.valid_emotions:
        #    raise ValueError(f"Invalid emotion value: {attributes['emotion']}")
        attributes['emotion'] = emotion
        return attributes

    def pinyin_conversion(self, py):
        # 将用户输入的拼音转化成tal的音素
        matches = re.findall(self.pinyin_pattern, py)
        if not matches:
            raise ValueError(f'非法拼音 {py}，未捕捉到拼音。')
        if len(matches[0]) != 2:
            raise ValueError(f'非法拼音，拼音组成不完整，缺失音调或者声/韵母 {py}。')
        else:
            pinyin, tone = matches[0]
            if not tal_cn_dict.get(pinyin):
                raise ValueError(f'非法拼音，拼音 {pinyin} 不在合法拼音集合中。')
            else:
                #pinyins = re.sub(r'[ ]', '', tal_cn_dict[pinyin])
                pinyins = tal_cn_dict[pinyin]
            if int(tone) >= 6:
                raise ValueError(f'音调错误，合法音调为1、2、3、4、5声。')
            # 六声变2声
            #tone = '2' if tone == '6' else tone
            if re.search(r'xer', pinyins):
                return [' '.join(pinyins.split(' ')[:-1])+tone, 'er2']
            else:
                return [pinyins+tone]
    
    def add_time(self, a, b):
        a_value = int(a.replace('ms', ''))
        b_value = int(b.replace('ms', ''))
        total = a_value + b_value
        return f'{total}ms'

    def get_contents_from_xml(self):
        '''返回 xml 内容，字符串和拼音的 list '''
        res = []
        for x1 in self.rnode:
            for x2 in x1.childNodes:
                if isinstance(x2, xml.dom.minidom.Text):
                    t = re.sub(r"\s+", ' ', x2.data).strip()
                    if t:
                        res.append(['text', '',t.strip()])
                elif x2.nodeName == 'break':
                    break_time = x2.getAttribute('time')
                    if res and res[-1][0] == 'break':
                        last_break_time = res[-1][-1]
                        sum_time = self.add_time(break_time, last_break_time)
                        res[-1][-1] = sum_time
                    else:
                        res.append(['break', '', break_time])
                elif x2.nodeName == 'math':
                    math_type = x2.getAttribute('interpret-as')
                    if math_type in {'latex', 'asciimath'}:
                        for x3 in x2.childNodes:
                            if isinstance(x3, xml.dom.minidom.Text):
                                res.append(['math', math_type, x3.data])
                    elif math_type == 'mathml':
                        tmp = ''.join(x3.toxml() for x3 in x2.childNodes if x3.nodeType == x3.ELEMENT_NODE)
                        tmp = re.sub(r'\s+', '', tmp)
                        res.append(['math', 'mathml', f'<math xmlns="http://www.w3.org/1998/Math/MathML">{tmp}</math>'])
                    else:
                        raise ValueError(f"非法数学公式格式: {math_type}")
                elif x2.nodeName == 'phoneme':
                    try:
                        lang = x2.getAttribute('lang')
                        phs = x2.getAttribute('ph').split('/')
                        x3 = x2.childNodes[0]
                    except:
                        raise ValueError(f'自定义读音元素phoneme属性缺失')
                    if lang == 'cn':
                        pinyin_value = []
                        for py in phs:
                            tal_py = self.pinyin_conversion(py)
                            pinyin_value += tal_py
                        #for x3 in x2.childNodes:
                        if isinstance(x3, xml.dom.minidom.Text):
                            c_text = x3.data
                            # c_text = re.findall(r'[\u4e00-\u9fff]', x3.data)
                            # if len(c_text) != len(pinyin_value):
                            #     raise ValueError(f'自定义拼音标签有误，文本序列长度和拼音序列长度不一致')
                            res.append(['pinyin', '/'.join(pinyin_value) + '/', c_text])
                    elif lang == 'en':
                        if isinstance(x3, xml.dom.minidom.Text):
                            c_text = x3.data.split(' ')
                        if len(c_text) != len(phs):
                            raise ValueError(f'自定义拼音标签有误，文本序列长度和拼音序列长度不一致')
                        res.append(['pinyin', ' / '.join(phs) + ' /', c_text])
                
                # 其他元素暂不处理，抛出ValueError
                else:
                    raise ValueError(f'非法ssml元素： {x2.nodeName}')
        return res
    
    def get_text(self):
        '''返回 xml 内容的所有文本内容的列表'''
        res = []

        for x1 in self.rnode:
            if x1.nodeType == Node.TEXT_NODE:
                res.append(x1.value)
            else:
                for x2 in x1.childNodes:
                    if isinstance(x2, xml.dom.minidom.Text):
                        res.append(x2.data)
                    else:
                        for x3 in x2.childNodes:
                            if isinstance(x3, xml.dom.minidom.Text):
                                res.append(x3.data)
                            else:
                                print("len(nodes of x3):", len(x3.childNodes))

        return res

    def get_xmlchild_list(self):
        '''返回 xml 内容的列表，包括所有文本内容(不带 tag)'''
        res = []

        for x1 in self.rnode:
            if x1.nodeType == Node.TEXT_NODE:
                res.append(x1.value)
            else:
                for x2 in x1.childNodes:
                    if isinstance(x2, xml.dom.minidom.Text):
                        res.append(x2.data)
                    else:
                        for x3 in x2.childNodes:
                            if isinstance(x3, xml.dom.minidom.Text):
                                res.append(x3.data)
                            else:
                                print("len(nodes of x3):", len(x3.childNodes))
        print(res)
        return res

    def get_all_tags(self, tag_name):
        '''获取所有的 tag 及属性值'''
        alltags = self.root.getElementsByTagName(tag_name)
        for x in alltags:
            if x.hasAttribute('pinyin'):  # pinyin
                print(x.tagName, 'pinyin',
                      x.getAttribute('pinyin'), x.firstChild.data)

    def get_text_and_sayas_tags(self):
        '''返回 xml 内容的列表，包括所有文本内容和<say-as> tag'''
        res = []

        for x1 in self.rnode:
            if x1.nodeType == Node.TEXT_NODE:
                res.append(x1.value)
            else:
                for x2 in x1.childNodes:
                    res.append(x2.toxml())
        return res
