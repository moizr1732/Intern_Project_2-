#!/usr/bin/env python3
"""
Test script for evaluation module
"""

import sys
sys.path.append('.')
from load_dataset import load_dataset
from utils.preprocessing import preprocess_dataset
from utils.feature_engineering import extract_features
from utils.training import train_models
from utils.evaluation import evaluate_models, select_best_model, print_detailed_report, create_comparison_table

def test_evaluation():
    """Test the evaluation pipeline"""
    print("🧪 Testing Model Evaluation...")
    
    # Load and prepare data
    df = load_dataset()
    if df is not None:
        df_processed = preprocess_dataset(df)
        X_train, X_test, y_train, y_test, vectorizer = extract_features(df_processed)
        
        # Train models
        models, performance = train_models(X_train, X_test, y_train, y_test)
        
        # Evaluate models
        results = evaluate_models(models, X_test, y_test)
        
        # Select best model
        best_name, best_f1 = select_best_model(results)
        
        # Print detailed report
        print_detailed_report(results, best_name)
        
        # Create comparison table
        comparison_df = create_comparison_table(results)
        
        print("🎉 Model evaluation complete!")
        return True
    return False

if __name__ == "__main__":
    test_evaluation()
