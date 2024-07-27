from pyvi import ViUtils, ViTokenizer
from googletrans import Translator as GoogleTranslator
import translate
from difflib import SequenceMatcher
from langdetect import detect
import underthesea

class Translation:
    def __init__(self, from_lang='vi', to_lang='en', mode='google'):
        # Initialize the Translation class with the chosen translation mode
        self.__mode = mode
        self.__from_lang = from_lang
        self.__to_lang = to_lang

        if mode == 'google':
            self.translator = GoogleTranslator()
        elif mode == 'translate':
            self.translator = translate.Translator(from_lang=from_lang, to_lang=to_lang)
        else:
            raise ValueError("Unsupported mode. Choose 'google' or 'translate'.")

    def preprocessing(self, text):
        # Preprocess text to lower case
        return text.lower()

    def __call__(self, text):
        text = self.preprocessing(text)
        if self.__mode == 'translate':
            return self.translator.translate(text)
        else:
            return self.translator.translate(text, dest=self.__to_lang).text

class Text_Preprocessing:
    def __init__(self, stopwords_path='./dict/vietnamese-stopwords-dash.txt'):
        # Load stopwords from file
        with open(stopwords_path, 'r', encoding='utf8') as f:
            lines = f.readlines()
        self.stop_words = [line.strip() for line in lines]

    def find_substring(self, string1, string2):
        # Find the longest common substring
        match = SequenceMatcher(None, string1, string2, autojunk=False).find_longest_match(0, len(string1), 0, len(string2))
        return string1[match.a:match.a + match.size].strip()

    def remove_stopwords(self, text):
        # Tokenize and remove stopwords from text
        text = ViTokenizer.tokenize(text)
        return " ".join([w for w in text.split() if w not in self.stop_words])

    def lowercasing(self, text):
        # Convert text to lower case
        return text.lower()

    def uppercasing(self, text):
        # Convert text to upper case
        return text.upper()

    def add_accents(self, text):
        # Add accents to text
        return ViUtils.add_accents(text)

    def remove_accents(self, text):
        # Remove accents from text
        return ViUtils.remove_accents(text)

    def sentence_segment(self, text):
        # Segment text into sentences
        return underthesea.sent_tokenize(text)

    def text_norm(self, text):
        # Normalize text
        return underthesea.text_normalize(text)

    def text_classify(self, text):
        # Classify text
        return underthesea.classify(text)

    def sentiment_analysis(self, text):
        # Perform sentiment analysis on text
        return underthesea.sentiment(text)

    def __call__(self, text):
        # Preprocess text and perform various text processing steps
        text = self.lowercasing(text)
        text = self.remove_stopwords(text)
        text = self.text_norm(text)
        categories = self.text_classify(text)
        return categories