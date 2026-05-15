#!/usr/bin/env python3
"""
Evaluation module for Resume Screening AI
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_models(models, X_test, y_test):
    """
    Evaluate all trained models
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test labels
        
    Returns:
        dict: Dictionary containing evaluation results for all models
    """
    results = {}
    
    print("📈 Evaluating Models...")
    
    for model_name, model in models.items():
        print(f"\n🔍 Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        # Store results
        results[model_name] = {
            'metrics': metrics,
            'predictions': y_pred,
            'classification_report': classification_report(y_test, y_pred, zero_division=0)
        }
        
        # Print metrics
        print(f"   Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall:    {metrics['recall']:.4f}")
        print(f"   F1-Score:  {metrics['f1_score']:.4f}")
    
    return results

def select_best_model(results):
    """
    Select the best model based on F1-score
    
    Args:
        results: Dictionary of evaluation results
        
    Returns:
        tuple: (best_model_name, best_f1_score)
    """
    best_model_name = max(results.keys(), key=lambda k: results[k]['metrics']['f1_score'])
    best_f1_score = results[best_model_name]['metrics']['f1_score']
    
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   F1-Score: {best_f1_score:.4f}")
    
    return best_model_name, best_f1_score

def print_detailed_report(results, best_model_name):
    """
    Print detailed evaluation report for the best model
    
    Args:
        results: Dictionary of evaluation results
        best_model_name: Name of the best model
    """
    print(f"\n📋 Detailed Report for {best_model_name}:")
    print("=" * 50)
    print(results[best_model_name]['classification_report'])

def create_comparison_table(results):
    """
    Create a comparison table of all models
    
    Args:
        results: Dictionary of evaluation results
        
    Returns:
        pd.DataFrame: Comparison table
    """
    comparison_data = []
    
    for model_name, result in results.items():
        metrics = result['metrics']
        comparison_data.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1_score']
        })
    
    df = pd.DataFrame(comparison_data)
    df = df.sort_values('F1-Score', ascending=False)
    
    print("\n📊 Model Comparison Table:")
    print(df.to_string(index=False, float_format='%.4f'))
    
    return df

def plot_confusion_matrix(y_true, y_pred, model_name):
    """
    Plot confusion matrix for a model
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name of the model
    """
    try:
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=sorted(set(y_true)), 
                   yticklabels=sorted(set(y_true)))
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(f'models/confusion_matrix_{model_name.replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Confusion matrix saved for {model_name}")
    except Exception as e:
        print(f"❌ Error creating confusion matrix: {e}")

if __name__ == "__main__":
    # Test the evaluation module
    import sys
    sys.path.append('..')
    from load_dataset import load_dataset
    from preprocessing import preprocess_dataset
    from feature_engineering import extract_features
    from training import train_models
    
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
        
        # Plot confusion matrix for best model
        plot_confusion_matrix(y_test, results[best_name]['predictions'], best_name)
        
        print("🎉 Model evaluation complete!")
