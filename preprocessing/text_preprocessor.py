import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import spacy
from typing import List

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')


class TextPreprocessor:
    def __init__(self, use_spacy: bool = False):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.use_spacy = use_spacy
        
        if use_spacy:
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                print("Spacy model not found. Run: python -m spacy download en_core_web_sm")
                self.nlp = None

    def remove_html(self, text: str) -> str:
        """Remove HTML tags from text"""
        clean = re.compile('<.*?>')
        return re.sub(clean, '', text)

    def remove_urls(self, text: str) -> str:
        """Remove URLs from text"""
        url_pattern = re.compile(r'http\S+|www\S+|https\S+')
        return url_pattern.sub('', text)

    def remove_emojis(self, text: str) -> str:
        """Remove emojis from text"""
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        return emoji_pattern.sub('', text)

    def to_lowercase(self, text: str) -> str:
        """Convert text to lowercase"""
        return text.lower()

    def remove_punctuation(self, text: str) -> str:
        """Remove punctuation from text"""
        return text.translate(str.maketrans('', '', string.punctuation))

    def remove_extra_spaces(self, text: str) -> str:
        """Remove extra spaces from text"""
        return ' '.join(text.split())

    def remove_numbers(self, text: str) -> str:
        """Remove numbers from text"""
        return re.sub(r'\d+', '', text)

    def lemmatize_text(self, text: str) -> str:
        """Lemmatize text using WordNetLemmatizer"""
        tokens = word_tokenize(text)
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(lemmatized_tokens)

    def remove_stopwords(self, text: str) -> str:
        """Remove stopwords from text"""
        tokens = word_tokenize(text)
        filtered_tokens = [token for token in tokens if token.lower() not in self.stop_words]
        return ' '.join(filtered_tokens)

    def preprocess(self, text: str, remove_stops: bool = True) -> str:
        """
        Apply all preprocessing steps to text
        """
        # Remove HTML
        text = self.remove_html(text)
        
        # Remove URLs
        text = self.remove_urls(text)
        
        # Remove emojis
        text = self.remove_emojis(text)
        
        # Convert to lowercase
        text = self.to_lowercase(text)
        
        # Remove punctuation
        text = self.remove_punctuation(text)
        
        # Remove numbers
        text = self.remove_numbers(text)
        
        # Remove extra spaces
        text = self.remove_extra_spaces(text)
        
        # Remove stopwords (optional)
        if remove_stops:
            text = self.remove_stopwords(text)
        
        # Lemmatize
        text = self.lemmatize_text(text)
        
        # Final space cleanup
        text = self.remove_extra_spaces(text)
        
        return text.strip()

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        tokens = word_tokenize(text)
        return tokens

    def preprocess_with_spacy(self, text: str) -> str:
        """Preprocess text using spaCy"""
        if self.nlp is None:
            return self.preprocess(text)
        
        doc = self.nlp(text)
        
        # Remove stopwords, punctuation, and lemmatize
        tokens = [
            token.lemma_.lower() 
            for token in doc 
            if not token.is_stop 
            and not token.is_punct 
            and not token.is_space
        ]
        
        return ' '.join(tokens)
