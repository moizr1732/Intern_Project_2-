#!/usr/bin/env python3
"""
Feature Engineering module for Resume Screening AI
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
import os

def extract_features(df):
    """
    Extract TF-IDF features from text data
    
    Args:
        df (pd.DataFrame): Dataset with cleaned_text column
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test, vectorizer)
    """
    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words='english'
    )
    
    # Prepare features and labels
    X = df['cleaned_text']
    y = df['category']
    
    # Split data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Fit and transform the data
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"🔢 Feature Extraction Complete!")
    print(f"Training set shape: {X_train_tfidf.shape}")
    print(f"Test set shape: {X_test_tfidf.shape}")
    print(f"Number of features: {len(vectorizer.get_feature_names_out())}")
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer

def save_vectorizer(vectorizer, filepath='models/vectorizer.pkl'):
    """
    Save the TF-IDF vectorizer
    
    Args:
        vectorizer: Trained TF-IDF vectorizer
        filepath (str): Path to save the vectorizer
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"✅ Vectorizer saved to {filepath}")

def load_vectorizer(filepath='models/vectorizer.pkl'):
    """
    Load the TF-IDF vectorizer
    
    Args:
        filepath (str): Path to the saved vectorizer
        
    Returns:
        vectorizer: Loaded TF-IDF vectorizer
    """
    try:
        with open(filepath, 'rb') as f:
            vectorizer = pickle.load(f)
        print(f"✅ Vectorizer loaded from {filepath}")
        return vectorizer
    except FileNotFoundError:
        print(f"❌ Vectorizer file not found at {filepath}")
        return None

if __name__ == "__main__":
    # Test the feature extraction
    import sys
    sys.path.append('..')
    from load_dataset import load_dataset
    from preprocessing import preprocess_dataset
    
    # Load and preprocess data
    df = load_dataset()
    if df is not None:
        df_processed = preprocess_dataset(df)
        
        # Extract features
        X_train, X_test, y_train, y_test, vectorizer = extract_features(df_processed)
        
        # Save vectorizer
        save_vectorizer(vectorizer)
        
        print("🎉 Feature engineering complete!")
