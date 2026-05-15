#!/usr/bin/env python3
"""
Test script for training module
"""

import sys
sys.path.append('.')
from load_dataset import load_dataset
from utils.preprocessing import preprocess_dataset
from utils.feature_engineering import extract_features
from utils.training import train_models, select_best_model, save_model

def test_training():
    """Test the training pipeline"""
    print("🧪 Testing Model Training...")
    
    # Load and prepare data
    df = load_dataset()
    if df is not None:
        df_processed = preprocess_dataset(df)
        X_train, X_test, y_train, y_test, vectorizer = extract_features(df_processed)
        
        # Train models
        models, performance = train_models(X_train, X_test, y_train, y_test)
        
        # Select and save best model
        best_name, best_model, best_perf = select_best_model(models, performance)
        save_model(best_model)
        
        print("🎉 Model training complete!")
        return True
    return False

if __name__ == "__main__":
    test_training()
