#!/usr/bin/env python3
"""
Test script for feature engineering module
"""

import sys
sys.path.append('.')
from load_dataset import load_dataset
from utils.preprocessing import preprocess_dataset
from utils.feature_engineering import extract_features, save_vectorizer

def test_feature_engineering():
    """Test the feature engineering pipeline"""
    print("🧪 Testing Feature Engineering...")
    
    # Load and preprocess data
    df = load_dataset()
    if df is not None:
        df_processed = preprocess_dataset(df)
        
        # Extract features
        X_train, X_test, y_train, y_test, vectorizer = extract_features(df_processed)
        
        # Save vectorizer
        save_vectorizer(vectorizer)
        
        print("🎉 Feature engineering complete!")
        return True
    return False

if __name__ == "__main__":
    test_feature_engineering()
