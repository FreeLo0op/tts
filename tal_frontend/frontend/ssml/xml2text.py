import os
# from xml_processor import DomXml
from tal_frontend.frontend.ssml.xml_processor import DomXml


def xml_reader(xml_file):
    if not xml_file.lower().endswith('.xml'):
        raise ValueError("The file must be a .xml file")

    try:
        with open(xml_file, 'r') as fin:
            xml_content = fin.read()
        dom_processor = DomXml(xml_content)
        speak_info = dom_processor.get_speak_info()
        contents = dom_processor.get_contents_from_xml()
        return contents, speak_info
    except FileNotFoundError:
        raise FileExistsError(f"The file {xml_file} was not found.")
    except ValueError as e:
        raise ValueError(f"ValueError: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}")


def xml_reader_string(xml_content):
    # if not xml_file.lower().endswith('.xml'):
    #     raise ValueError("The file must be a .xml file")
    try:
        # with open(xml_file, 'r') as fin:
        #     xml_content = fin.read()
        dom_processor = DomXml(xml_content)
        speak_info = dom_processor.get_speak_info()
        contents = dom_processor.get_contents_from_xml()
        return contents, speak_info
    # except FileNotFoundError:
    #     raise FileExistsError(f"The file {xml_file} was not found.")
    except ValueError as e:
        raise ValueError(f"ValueError: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}")

# if __name__ == '__main__':
#    file = r'/mnt/cfs/SPEECH/hupeng/git_loc_workspace/tal_frontend/tal_frontend/frontend/ssml/test_data.xml'
#    contents = xml_reader(file)
#    print(contents)
