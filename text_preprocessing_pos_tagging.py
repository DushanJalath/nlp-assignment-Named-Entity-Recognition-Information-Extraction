import nltk
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
import re
from collections import Counter

class TextPreprocessor:
    """
    Handles text cleaning, tokenization, and POS tagging
    Responsibilities: Person 1
    """
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """
        Clean the input text by removing unwanted characters
        """
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:\-\']', '', text)
        return text.strip()
    
    def tokenize_sentences(self, text):
        """
        Split text into sentences
        """
        sentences = nltk.sent_tokenize(text)
        return sentences
    
    def tokenize_words(self, text):
        """
        Split text into words/tokens
        """
        tokens = word_tokenize(text)
        return tokens
    
    def pos_tagging(self, tokens):
        """
        Perform Part-of-Speech tagging on tokens
        Returns: List of (word, POS_tag) tuples
        """
        pos_tags = pos_tag(tokens)
        return pos_tags
    
    def preprocess_pipeline(self, text):
        """
        Complete preprocessing pipeline
        Returns: Dictionary with preprocessed data
        """
        # Step 1: Clean text
        cleaned_text = self.clean_text(text)
        
        # Step 2: Tokenize into sentences
        sentences = self.tokenize_sentences(cleaned_text)
        
        # Step 3: Tokenize into words
        tokens = self.tokenize_words(cleaned_text)
        
        # Step 4: POS Tagging
        pos_tags = self.pos_tagging(tokens)
        
        # Step 5: Filter out stopwords (optional, for analysis)
        filtered_tokens = [word for word in tokens 
                          if word.lower() not in self.stop_words 
                          and word.isalnum()]
        
        return {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'sentences': sentences,
            'tokens': tokens,
            'pos_tags': pos_tags,
            'filtered_tokens': filtered_tokens
        }
    
    def display_pos_stats(self, pos_tags):
        """
        Display statistics about POS tags
        """
        pos_counts = Counter([tag for word, tag in pos_tags])
        print("\n" + "="*60)
        print("PERSON 1 OUTPUT: POS TAGGING STATISTICS")
        print("="*60)
        print(f"Total tokens: {len(pos_tags)}")
        print(f"\nTop 10 POS tags:")
        for tag, count in pos_counts.most_common(10):
            print(f"  {tag}: {count}")
        return pos_counts