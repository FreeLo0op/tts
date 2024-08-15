from abc import ABC
from abc import abstractmethod
from g2p_en import G2p
from tal_frontend.frontend.g2p.vocab import Vocab

class Phonetics(ABC):
    @abstractmethod
    def __call__(self, sentence):
        pass

    @abstractmethod
    def phoneticize(self, sentence):
        pass

    @abstractmethod
    def numericalize(self, phonemes):
        pass


class English(Phonetics):
    """ Normalize the input text sequence and convert into pronunciation id sequence.

    https://github.com/Kyubyong/g2p/blob/master/g2p_en/g2p.py

    phonemes = ["<pad>", "<unk>", "<s>", "</s>"] + [   
        'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0',
        'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH',
        'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1',
        'EY2', 'F', 'G', 'HH',
        'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L',
        'M', 'N', 'NG', 'OW0', 'OW1',
        'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH',
        'UH0', 'UH1', 'UH2', 'UW',
        'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']
    """

    LEXICON = {
        # key using lowercase
        "AI".lower(): [["EY0", "AY1"]],
    }

    def __init__(self, phone_vocab_path=None):
        self.backend = G2p()
        self.backend.cmu.update(English.LEXICON)
        self.phonemes = list(self.backend.phonemes)
        self.punctuations = [" ","-","...",",",".","?","!",]
        self.vocab = Vocab(self.phonemes + self.punctuations)
        self.vocab_phones = {}
        self.punc = "、：，；。？！“”‘’':,;.?!"
        if phone_vocab_path:
            with open(phone_vocab_path, 'rt', encoding='utf-8') as f:
                phn_id = [line.strip().split() for line in f.readlines()]
            for phn, id in phn_id:
                self.vocab_phones[phn] = int(id)

    def phoneticize(self, sentence):
        """ Normalize the input text sequence and convert it into pronunciation sequence.
        Args:
            sentence (str): The input text sequence.
        Returns: 
            List[str]: The list of pronunciation sequence.
        """
        start = self.vocab.start_symbol
        end = self.vocab.end_symbol
        phonemes = ([] if start is None else [start]) \
                   + self.backend(sentence) \
                   + ([] if end is None else [end])
        phonemes = [item for item in phonemes if item in self.vocab.stoi]
        return ' '.join(phonemes[1:-1])
    

    def numericalize(self, phonemes):
        """ Convert pronunciation sequence into pronunciation id sequence.
        Args:
            phonemes (List[str]): The list of pronunciation sequence.
        Returns: 
            List[int]: The list of pronunciation id sequence.
        """
        ids = [
            self.vocab.lookup(item) for item in phonemes
            if item in self.vocab.stoi
        ]
        return ids

    def reverse(self, ids):
        """ Reverse the list of pronunciation id sequence to a list of pronunciation sequence.
        Args:
            ids (List[int]): The list of pronunciation id sequence.
        Returns: 
            List[str]: The list of pronunciation sequence.
        """
        return [self.vocab.reverse(i) for i in ids]

    def __call__(self, sentence):
        """ Convert the input text sequence into pronunciation id sequence.
        Args:
            sentence(str): The input text sequence.
        Returns: 
            List[str]: The list of pronunciation id sequence.
        """
        return self.numericalize(self.phoneticize(sentence))

    @property
    def vocab_size(self):
        """ Vocab size.
        """
        return len(self.vocab)