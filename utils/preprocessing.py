#!/usr/bin/env python3
"""
Preprocessing module for Resume Screening AI
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data (only needed once)
try:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

def clean_text(text):
    """
    Clean and preprocess text data
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned text
        
    Raises:
        ValueError: If input is empty or not a string
    """
    # Error handling
    if not text:
        raise ValueError("Input text cannot be empty")
    
    if not isinstance(text, str):
        raise ValueError("Input must be a string")
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    
    # Remove extra spaces and join
    cleaned_text = ' '.join(tokens)
    
    # Remove extra whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text

def preprocess_dataset(df):
    """
    Preprocess entire dataset
    
    Args:
        df (pd.DataFrame): Dataset with resume_text column
        
    Returns:
        pd.DataFrame: Dataset with cleaned text
    """
    df = df.copy()
    df['cleaned_text'] = df['resume_text'].apply(clean_text)
    return df

if __name__ == "__main__":
    # Test the function
    test_text = "Experienced data scientist with 5 years of expertise in machine learning, statistical analysis, and data visualization."
    print("Original text:", test_text)
    print("Cleaned text:", clean_text(test_text))
