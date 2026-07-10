import re
import nltk

class TextCleaner:
    def __init__(self):
        self.use_nltk = True
        try:
            # Ensure NLTK packages are downloaded
            try:
                nltk.data.find("corpora/stopwords")
            except LookupError:
                nltk.download("stopwords", quiet=True)

            try:
                nltk.data.find("corpora/wordnet")
            except LookupError:
                nltk.download("wordnet", quiet=True)

            try:
                nltk.data.find("tokenizers/punkt")
            except LookupError:
                nltk.download("punkt", quiet=True)

            try:
                nltk.data.find("tokenizers/punkt_tab")
            except LookupError:
                nltk.download("punkt_tab", quiet=True)

            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer
            from nltk.tokenize import word_tokenize
            
            self.stop_words = set(stopwords.words("english"))
            self.lemmatizer = WordNetLemmatizer()
            self.word_tokenize = word_tokenize
        except Exception as e:
            print(f"Warning: NLTK initialization failed ({e}). Falling back to regex tokenizer.")
            self.use_nltk = False
            self.stop_words = {"the", "a", "an", "and", "or", "but", "if", "because", "as", "what", "which", "this", "that", "these", "those", "then", "just", "so", "for", "with", "about", "into", "of", "to", "by", "from", "on", "in", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "not"}

    def clean(self, text: str, lemmatize: bool = True) -> str:
        if not text or not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
        
        # Lowercase
        text = text.lower()
        
        # Remove non-alphanumeric characters except basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?;:\']', ' ', text)
        
        if self.use_nltk:
            try:
                tokens = self.word_tokenize(text)
                cleaned_tokens = []
                for word in tokens:
                    w = word.strip(".,!?;:\'")
                    if w and w not in self.stop_words and len(w) > 1:
                        if lemmatize:
                            w = self.lemmatizer.lemmatize(w)
                        cleaned_tokens.append(w)
                return " ".join(cleaned_tokens)
            except Exception:
                pass # fall back to regex tokenization
                
        # Simple regex tokenization fallback
        tokens = re.findall(r'\b[a-zA-Z0-9\']+\b', text)
        cleaned_tokens = [w for w in tokens if w not in self.stop_words and len(w) > 1]
        return " ".join(cleaned_tokens)
