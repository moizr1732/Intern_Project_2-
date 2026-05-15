#!/usr/bin/env python3
"""
Final validation script for Resume Screening AI System
"""

import sys
import os
sys.path.append('.')

def test_complete_system():
    """Test all components of the system"""
    print("🧪 FINAL SYSTEM VALIDATION")
    print("=" * 50)
    
    # Test 1: Dataset Loading
    print("\n📊 Test 1: Dataset Loading...")
    try:
        from load_dataset import load_dataset
        df = load_dataset()
        assert df is not None, "Dataset loading failed"
        assert len(df) > 0, "Dataset is empty"
        assert 'resume_text' in df.columns, "Missing resume_text column"
        assert 'category' in df.columns, "Missing category column"
        print("✅ Dataset loading: PASSED")
    except Exception as e:
        print(f"❌ Dataset loading: FAILED - {e}")
        return False
    
    # Test 2: Preprocessing
    print("\n🧹 Test 2: Preprocessing...")
    try:
        from utils.preprocessing import clean_text, preprocess_dataset
        test_text = "Experienced data scientist with Python skills!"
        cleaned = clean_text(test_text)
        assert isinstance(cleaned, str), "Clean text should be string"
        assert len(cleaned) > 0, "Clean text should not be empty"
        
        df_processed = preprocess_dataset(df)
        assert 'cleaned_text' in df_processed.columns, "Missing cleaned_text column"
        print("✅ Preprocessing: PASSED")
    except Exception as e:
        print(f"❌ Preprocessing: FAILED - {e}")
        return False
    
    # Test 3: Feature Engineering
    print("\n🔢 Test 3: Feature Engineering...")
    try:
        from utils.feature_engineering import extract_features, load_vectorizer
        X_train, X_test, y_train, y_test, vectorizer = extract_features(df_processed)
        assert X_train.shape[0] > 0, "Training features should not be empty"
        assert X_test.shape[0] > 0, "Test features should not be empty"
        assert vectorizer is not None, "Vectorizer should not be None"
        
        # Test loading saved vectorizer
        loaded_vectorizer = load_vectorizer()
        assert loaded_vectorizer is not None, "Could not load saved vectorizer"
        print("✅ Feature Engineering: PASSED")
    except Exception as e:
        print(f"❌ Feature Engineering: FAILED - {e}")
        return False
    
    # Test 4: Model Training
    print("\n🤖 Test 4: Model Training...")
    try:
        from utils.training import train_models, load_model
        models, performance = train_models(X_train, X_test, y_train, y_test)
        assert len(models) > 0, "No models trained"
        
        # Test loading saved model
        loaded_model = load_model()
        assert loaded_model is not None, "Could not load saved model"
        print("✅ Model Training: PASSED")
    except Exception as e:
        print(f"❌ Model Training: FAILED - {e}")
        return False
    
    # Test 5: Ranking System
    print("\n🧮 Test 5: Ranking System...")
    try:
        from utils.ranking import calculate_similarity, rank_resumes, get_match_score
        job_desc = "Looking for data scientist with Python"
        resume_text = "Experienced data scientist with Python and machine learning"
        
        similarity = calculate_similarity(job_desc, resume_text, loaded_vectorizer)
        assert 0 <= similarity <= 1, "Similarity should be between 0 and 1"
        
        match_score = get_match_score(similarity)
        assert 0 <= match_score <= 100, "Match score should be between 0 and 100"
        
        # Test ranking
        ranked = rank_resumes(job_desc, df_processed.head(5), loaded_vectorizer)
        assert len(ranked) > 0, "Ranking should return results"
        print("✅ Ranking System: PASSED")
    except Exception as e:
        print(f"❌ Ranking System: FAILED - {e}")
        return False
    
    # Test 6: Complete Pipeline
    print("\n🔗 Test 6: Complete Pipeline...")
    try:
        # Test complete resume analysis
        test_resume = """
        Experienced data scientist with 5 years of expertise in machine learning, 
        statistical analysis, and data visualization. Proficient in Python, R, SQL, 
        and various ML frameworks including TensorFlow and PyTorch.
        """
        
        test_job = "Looking for a data scientist with Python and machine learning experience"
        
        # Preprocess
        cleaned_resume = clean_text(test_resume)
        
        # Predict
        resume_vector = loaded_vectorizer.transform([cleaned_resume])
        prediction = loaded_model.predict(resume_vector)[0]
        
        # Calculate similarity
        similarity = calculate_similarity(test_job, cleaned_resume, loaded_vectorizer)
        match_score = get_match_score(similarity)
        
        assert prediction is not None, "Prediction should not be None"
        assert isinstance(prediction, str), "Prediction should be string"
        assert match_score >= 0, "Match score should be non-negative"
        
        print(f"   Predicted Role: {prediction}")
        print(f"   Match Score: {match_score:.1f}%")
        print("✅ Complete Pipeline: PASSED")
    except Exception as e:
        print(f"❌ Complete Pipeline: FAILED - {e}")
        return False
    
    # Test 7: File Structure
    print("\n📁 Test 7: File Structure...")
    required_files = [
        'data/resumes.csv',
        'utils/preprocessing.py',
        'utils/feature_engineering.py',
        'utils/training.py',
        'utils/evaluation.py',
        'utils/ranking.py',
        'app/streamlit_app.py',
        'main.py',
        'requirements.txt',
        'models/model.pkl',
        'models/vectorizer.pkl'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ File Structure: FAILED - Missing files: {missing_files}")
        return False
    
    print("✅ File Structure: PASSED")
    
    # Final Summary
    print("\n🎉 ALL TESTS PASSED!")
    print("=" * 50)
    print("✅ Resume Screening AI System is ready!")
    print("✅ All components are working correctly")
    print("✅ Models trained and saved")
    print("✅ UI ready for use")
    
    print("\n🚀 To start the application:")
    print("   streamlit run app/streamlit_app.py")
    
    return True

if __name__ == "__main__":
    success = test_complete_system()
    if success:
        print("\n🎊 System validation completed successfully!")
    else:
        print("\n❌ System validation failed. Please check the errors above.")
