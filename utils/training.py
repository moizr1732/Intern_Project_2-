#!/usr/bin/env python3
"""
Training module for Resume Screening AI
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import os

def train_models(X_train, X_test, y_train, y_test):
    """
    Train multiple classification models
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training labels
        y_test: Test labels
        
    Returns:
        dict: Dictionary containing trained models and their performance
    """
    models = {}
    performance = {}
    
    print("🤖 Training Models...")
    
    # 1. Logistic Regression
    print("   Training Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    
    models['Logistic Regression'] = lr_model
    performance['Logistic Regression'] = calculate_metrics(y_test, lr_pred)
    
    # 2. Linear SVM
    print("   Training Linear SVM...")
    svm_model = SVC(kernel='linear', random_state=42, probability=True)
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    
    models['Linear SVM'] = svm_model
    performance['Linear SVM'] = calculate_metrics(y_test, svm_pred)
    
    # 3. Random Forest
    print("   Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    models['Random Forest'] = rf_model
    performance['Random Forest'] = calculate_metrics(y_test, rf_pred)
    
    # Print performance
    print("\n📊 Model Performance:")
    for model_name, metrics in performance.items():
        print(f"\n{model_name}:")
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1-Score:  {metrics['f1_score']:.4f}")
    
    return models, performance

def calculate_metrics(y_true, y_pred):
    """
    Calculate classification metrics
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        dict: Dictionary of metrics
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

def select_best_model(models, performance):
    """
    Select the best model based on F1-score
    
    Args:
        models: Dictionary of trained models
        performance: Dictionary of performance metrics
        
    Returns:
        tuple: (best_model_name, best_model, best_performance)
    """
    best_model_name = max(performance.keys(), key=lambda k: performance[k]['f1_score'])
    best_model = models[best_model_name]
    best_performance = performance[best_model_name]
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   F1-Score: {best_performance['f1_score']:.4f}")
    
    return best_model_name, best_model, best_performance

def save_model(model, filepath='models/model.pkl'):
    """
    Save the trained model
    
    Args:
        model: Trained model
        filepath (str): Path to save the model
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"✅ Model saved to {filepath}")

def load_model(filepath='models/model.pkl'):
    """
    Load the trained model
    
    Args:
        filepath (str): Path to the saved model
        
    Returns:
        model: Loaded model
    """
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Model loaded from {filepath}")
        return model
    except FileNotFoundError:
        print(f"❌ Model file not found at {filepath}")
        return None

if __name__ == "__main__":
    # Test the training module
    import sys
    sys.path.append('..')
    from load_dataset import load_dataset
    from preprocessing import preprocess_dataset
    from feature_engineering import extract_features
    
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
