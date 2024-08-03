from pyvi import ViUtils, ViTokenizer
import googletrans
import translate
from difflib import SequenceMatcher
from langdetect import detect
import underthesea
from transformers import BertModel, BertTokenizer
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Translation():
    def __init__(self, from_lang='vi', to_lang='en', mode='google'):
        # The class Translation is a wrapper for the two translation libraries, googletrans and translate. 
        self.__mode = mode
        self.__from_lang = from_lang
        self.__to_lang = to_lang

        if mode in 'googletrans':
            self.translator = googletrans.Translator()
        elif mode in 'translate':
            self.translator = translate.Translator(from_lang=from_lang,to_lang=to_lang)

    def preprocessing(self, text):

        return text.lower()

    def __call__(self, text):

        text = self.preprocessing(text)
        return self.translator.translate(text) if self.__mode in 'translate' \
                else self.translator.translate(text, dest=self.__to_lang).text

class Text_Preprocessing():
    def __init__(self, stopwords_path='./dict/vietnamese-stopwords-dash.txt'):
        with open(stopwords_path, 'rb') as f:
            lines = f.readlines()
        self.stop_words = [line.decode('utf8').replace('\n','') for line in lines]

    def find_substring(self, string1, string2):

        match = SequenceMatcher(None, string1, string2, autojunk=False).find_longest_match(0, len(string1), 0, len(string2))
        return string1[match.a:match.a + match.size].strip()

    def remove_stopwords(self, text):

        text = ViTokenizer.tokenize(text)
        return " ".join([w for w in text.split() if w not in self.stop_words])

    def lowercasing(self, text):
        return text.lower() 

    def uppercasing(self, text):
        return text.upper()

    def add_accents(self, text): 

        return ViUtils.add_accents(u"{}".format(text))

    def remove_accents(self, text): 

        return ViUtils.remove_accents(u"{}".format(text))

    def sentence_segment(self, text):

        return underthesea.sent_tokenize(text)

    def text_norm(self, text):

        return underthesea.text_normalize(text)  

    def text_classify(self, text):

        return underthesea.classify(text)

    def sentiment_analysis(self, text):

        return underthesea.sentiment(text)

    def __call__(self, text):

        text = self.lowercasing(text)
        text = self.remove_stopwords(text)
        # text = self.remove_accents(text)
        # text = self.add_accents(text)
        text = self.text_norm(text)
        categories = self.text_classify(text)
        return categories


class TextProcessingWithFeature():
    def __init__(self, stopwords_path='./dict/vietnamese-stopwords-dash.txt', translation_mode='google'):
        self.text_preprocessor = Text_Preprocessing(stopwords_path=stopwords_path)
        self.translator = Translation(mode=translation_mode)
        
        # Khởi tạo BERT tokenizer và model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')

    def encode_text(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', max_length=512, truncation=True, padding='max_length')
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

    def process_text(self, text):
        # Tiền xử lý văn bản
        preprocessed_text = self.text_preprocessor(text)
        
        # Phân tích cảm xúc
        sentiment = self.text_preprocessor.sentiment_analysis(preprocessed_text)
        
        # Dịch văn bản (nếu cần)
        translated_text = self.translator(preprocessed_text)
        
        # Mã hóa văn bản bằng BERT
        embedding = self.encode_text(preprocessed_text)
        
        return {
            'preprocessed_text': preprocessed_text,
            'sentiment': sentiment,
            'translated_text': translated_text,
            'embedding': embedding
        }

    def __call__(self, text):
        return self.process_text(text)
    
    
