#!/usr/bin/env python3
"""
Main pipeline for Resume Screening AI
"""

import os
import sys
from load_dataset import load_dataset
from utils.preprocessing import preprocess_dataset
from utils.feature_engineering import extract_features, save_vectorizer
from utils.training import train_models, select_best_model, save_model
from utils.evaluation import evaluate_models, select_best_model as eval_best_model, create_comparison_table
from utils.ranking import calculate_similarity, rank_resumes, get_match_score, explain_match

def main():
    """Main pipeline function"""
    print("🚀 Starting Resume Screening AI Pipeline...")
    print("=" * 50)
    
    # Step 1: Load Dataset
    print("\n📊 STEP 1: Loading Dataset...")
    df = load_dataset()
    if df is None:
        print("❌ Failed to load dataset. Exiting...")
        return False
    
    # Step 2: Preprocess Data
    print("\n🧹 STEP 2: Preprocessing Data...")
    df_processed = preprocess_dataset(df)
    print(f"✅ Preprocessed {len(df_processed)} resumes")
    
    # Step 3: Feature Engineering
    print("\n🔢 STEP 3: Feature Engineering...")
    X_train, X_test, y_train, y_test, vectorizer = extract_features(df_processed)
    
    # Save vectorizer
    save_vectorizer(vectorizer)
    
    # Step 4: Model Training
    print("\n🤖 STEP 4: Training Models...")
    models, performance = train_models(X_train, X_test, y_train, y_test)
    
    # Step 5: Model Evaluation
    print("\n📈 STEP 5: Evaluating Models...")
    results = evaluate_models(models, X_test, y_test)
    
    # Select best model
    best_name, best_f1 = eval_best_model(results)
    
    # Create comparison table
    comparison_df = create_comparison_table(results)
    
    # Step 6: Save Best Model
    print("\n💾 STEP 6: Saving Best Model...")
    best_model = models[best_name]
    save_model(best_model)
    
    # Step 7: Test Ranking System
    print("\n🧮 STEP 7: Testing Ranking System...")
    
    # Test with sample job description
    sample_job_desc = "Looking for a data scientist with Python, machine learning, and statistical analysis experience."
    sample_resume = df_processed['resume_text'].iloc[0]
    
    # Calculate similarity
    similarity = calculate_similarity(sample_job_desc, sample_resume, vectorizer)
    print(f"Sample similarity score: {similarity:.4f}")
    print(f"Match percentage: {get_match_score(similarity):.1f}%")
    
    # Get explanation
    explanation = explain_match(sample_job_desc, sample_resume, vectorizer)
    print(f"Common features found: {explanation['total_common_features']}")
    
    # Step 8: Final Summary
    print("\n🎉 PIPELINE COMPLETE!")
    print("=" * 50)
    print(f"Dataset: {len(df_processed)} resumes")
    print(f"Categories: {df_processed['category'].nunique()}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Best Model: {best_name}")
    print(f"Best F1-Score: {best_f1:.4f}")
    print(f"Model saved: models/model.pkl")
    print(f"Vectorizer saved: models/vectorizer.pkl")
    
    print("\n✅ Ready to use with Streamlit UI!")
    print("Run: streamlit run app/streamlit_app.py")
    
    return True

def test_complete_system():
    """Test the complete system with a sample resume"""
    print("\n🧪 TESTING COMPLETE SYSTEM...")
    print("=" * 30)
    
    try:
        # Load saved components
        from utils.feature_engineering import load_vectorizer
        from utils.training import load_model
        
        vectorizer = load_vectorizer()
        model = load_model()
        
        if vectorizer is None or model is None:
            print("❌ Could not load saved components")
            return False
        
        # Test data
        test_resume = """
        Experienced data scientist with 5 years of expertise in machine learning, 
        statistical analysis, and data visualization. Proficient in Python, R, SQL, 
        and various ML frameworks including TensorFlow and PyTorch.
        """
        
        test_job_desc = "Looking for a data scientist with Python and machine learning experience."
        
        # Preprocess
        from utils.preprocessing import clean_text
        cleaned_resume = clean_text(test_resume)
        
        # Predict category
        resume_vector = vectorizer.transform([cleaned_resume])
        prediction = model.predict(resume_vector)[0]
        
        # Calculate similarity
        similarity = calculate_similarity(test_job_desc, cleaned_resume, vectorizer)
        
        print(f"📄 Resume Category Prediction: {prediction}")
        print(f"🎯 Job Match Score: {get_match_score(similarity):.1f}%")
        
        # Get explanation
        explanation = explain_match(test_job_desc, cleaned_resume, vectorizer)
        print(f"🔍 Top Matching Keywords: {[k['feature'] for k in explanation['top_keywords'][:3]]}")
        
        print("✅ System test passed!")
        return True
        
    except Exception as e:
        print(f"❌ System test failed: {e}")
        return False

if __name__ == "__main__":
    # Run main pipeline
    success = main()
    
    if success:
        # Test the complete system
        test_complete_system()
        
        print("\n🚀 Resume Screening AI is ready!")
        print("Start the UI with: streamlit run app/streamlit_app.py")
    else:
        print("\n❌ Pipeline failed. Please check the errors above.")
