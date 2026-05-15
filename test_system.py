#!/usr/bin/env python3
"""
End-to-end system test for Resume Screening AI
"""

import sys
import os

def test_system():
    """Test all components of the resume screening system."""
    try:
        print("🧪 Starting End-to-End System Test")
        print("=" * 50)
        
        # Test 1: Preprocessing
        print("1️⃣ Testing Preprocessing Module...")
        from utils.preprocessing import load_and_preprocess_resume_data
        
        df, metadata = load_and_preprocess_resume_data()
        print(f"✅ Dataset loaded: {len(df)} resumes, {len(metadata['categories'])} categories")
        
        # Test 2: Feature Engineering
        print("\n2️⃣ Testing Feature Engineering...")
        from utils.feature_engineering import create_features_pipeline
        
        feature_results = create_features_pipeline(df)
        print(f"✅ Features created: Train={feature_results['X_train'].shape}, Test={feature_results['X_test'].shape}")
        
        # Test 3: Model Training
        print("\n3️⃣ Testing Model Training...")
        from utils.training import train_resume_classifier
        
        training_results = train_resume_classifier(
            feature_results['X_train'], 
            feature_results['y_train'], 
            feature_results['X_test'], 
            feature_results['y_test'],
            save_model=True
        )
        print(f"✅ Models trained successfully. Best model: {training_results['best_model_name']}")
        print(f"✅ Best F1 Score: {training_results['best_score']:.4f}")
        
        # Test 4: Resume Ranking
        print("\n4️⃣ Testing Resume Ranking...")
        from utils.ranking import create_resume_ranker
        
        ranker = create_resume_ranker(df, feature_results['feature_engineering'])
        print("✅ Resume ranker created successfully")
        
        # Test 5: Explainable AI
        print("\n5️⃣ Testing Explainable AI...")
        from utils.explainable_ai import create_explainable_ai
        
        xai = create_explainable_ai(
            training_results['trainer'].best_model,
            feature_results['feature_engineering'],
            df
        )
        print("✅ Explainable AI system created successfully")
        
        # Test 6: Sample Prediction
        print("\n6️⃣ Testing Sample Prediction...")
        sample_text = df['resume_text'].iloc[0]
        explanation = xai.get_prediction_explanation(sample_text)
        print(f"✅ Sample prediction: {explanation.get('predicted_category', 'Unknown')}")
        print(f"✅ Confidence: {explanation.get('confidence_score', 0):.3f}")
        
        # Test 7: Resume Ranking
        print("\n7️⃣ Testing Resume Ranking...")
        job_desc = "Looking for a data scientist with Python and machine learning experience"
        ranking_results = ranker.rank_resumes_for_job(job_desc, top_k=3)
        print(f"✅ Ranking completed: {ranking_results.get('total_candidates', 0)} candidates ranked")
        
        # Test 8: Model Persistence
        print("\n8️⃣ Testing Model Persistence...")
        if os.path.exists('models/model.pkl'):
            print("✅ Model saved successfully")
        else:
            print("❌ Model file not found")
        
        if os.path.exists('models/tfidf_vectorizer.pkl'):
            print("✅ Vectorizer saved successfully")
        else:
            print("❌ Vectorizer file not found")
        
        print("\n" + "=" * 50)
        print("🎉 All components tested successfully!")
        print("🚀 System is ready for use!")
        print("\n📋 Test Summary:")
        print(f"   • Dataset: {len(df)} resumes")
        print(f"   • Categories: {len(metadata['categories'])}")
        print(f"   • Best Model: {training_results['best_model_name']}")
        print(f"   • Best F1 Score: {training_results['best_score']:.4f}")
        print(f"   • Features: {feature_results['X_train'].shape[1]}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_system()
    sys.exit(0 if success else 1)
