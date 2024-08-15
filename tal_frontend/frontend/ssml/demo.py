from xml_processor import DomXml

def main():
    xml_file = r'/mnt/cfs/SPEECH/hupeng/git_loc_workspace/tal_frontend/tal_frontend/frontend/ssml/test_data.xml'
    # 测试数据：包含 XML 和普通文本

    test_string = '''
    This is regular text.
    <speak>
        Hello, how are you?
        <say-as pinyin="nǐ hǎo">你好</say-as>
    </speak>
    This is more text following XML.
    '''

    '''
    # 创建 MixTextProcessor 实例
    processor = MixTextProcessor()

    # 获取分割后的内容
    split_content = processor.get_content_split(test_string)
    print("Split Content:")
    for part in split_content:
        print(part)
    '''
    
    # 使用 DomXml 处理 XML 字符串
    #xml_content = '''
    #<speak>
    #    Hello, how are you?
    #    <say-as pinyin="ni7 lou7">你好</say-as>
    #</speak>
    #'''

    with open(xml_file, 'r') as fin:
        xml_content = fin.read()
    dom_processor = DomXml(xml_content)

    # 获取文本内容
    #text_content = dom_processor.get_text()
    #print(text_content)

    # 获取带拼音的内容
    pinyin_content = dom_processor.get_pinyins_for_xml()
    print("\nContent from XML:")
    for item in pinyin_content:
        print(item)

if __name__ == "__main__":
    main()