#!/usr/bin/env python3
"""
Load and validate dataset for Resume Screening AI
"""

import pandas as pd
import numpy as np

def load_dataset():
    """Load and validate the dataset"""
    try:
        # Load dataset
        df = pd.read_csv('data/resumes.csv')
        
        # Print basic info
        print("📊 Dataset Loaded Successfully!")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check required columns
        required_columns = ['resume_text', 'category']
        if not all(col in df.columns for col in required_columns):
            print(f"❌ Missing required columns. Found: {list(df.columns)}")
            return None
        
        # Show first 5 rows
        print("\n📋 First 5 rows:")
        print(df.head())
        
        # Check for missing values
        print(f"\n🔍 Missing values:")
        print(df.isnull().sum())
        
        # Check for empty rows
        empty_text = df['resume_text'].str.strip().eq('').sum()
        print(f"\n⚠️ Empty resume_text rows: {empty_text}")
        
        # Remove rows with missing or empty text
        df = df.dropna(subset=['resume_text'])
        df = df[df['resume_text'].str.strip() != '']
        
        print(f"\n✅ Clean dataset shape: {df.shape}")
        print(f"📊 Categories: {df['category'].value_counts()}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return None

if __name__ == "__main__":
    dataset = load_dataset()
    if dataset is not None:
        print("🎉 Dataset ready for processing!")
