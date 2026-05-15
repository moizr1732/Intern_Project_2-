#!/usr/bin/env python3
"""
Test script for ranking system
"""

import sys
sys.path.append('.')
from load_dataset import load_dataset
from utils.preprocessing import preprocess_dataset
from utils.feature_engineering import load_vectorizer
from utils.ranking import calculate_similarity, rank_resumes, get_match_score, explain_match

def test_ranking():
    """Test the ranking system"""
    print("🧪 Testing Resume Ranking...")
    
    # Load data
    df = load_dataset()
    if df is not None:
        df_processed = preprocess_dataset(df)
        
        # Load vectorizer
        vectorizer = load_vectorizer()
        if vectorizer is not None:
            # Test job description
            job_desc = "Looking for a data scientist with Python, machine learning, and statistical analysis experience."
            
            # Test resume
            test_resume = df_processed['resume_text'].iloc[0]
            
            # Calculate similarity
            similarity = calculate_similarity(job_desc, test_resume, vectorizer)
            print(f"Similarity score: {similarity:.4f}")
            print(f"Match percentage: {get_match_score(similarity):.1f}%")
            
            # Rank all resumes
            ranked_resumes = rank_resumes(job_desc, df_processed, vectorizer, top_k=3)
            
            print("\n🏆 Top 3 Matching Resumes:")
            for i, resume in enumerate(ranked_resumes, 1):
                print(f"{i}. {resume['category']} - Score: {resume['similarity_score']:.4f}")
            
            # Get explanation
            explanation = explain_match(job_desc, test_resume, vectorizer)
            print(f"\n🔍 Explanation:")
            print(f"Common features found: {explanation['total_common_features']}")
            print("Top matching keywords:")
            for keyword in explanation['top_keywords']:
                print(f"  - {keyword['feature']}")
            
            print("🎉 Ranking system test complete!")
            return True
    return False

if __name__ == "__main__":
    test_ranking()
