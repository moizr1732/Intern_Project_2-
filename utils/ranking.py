#!/usr/bin/env python3
"""
Ranking system for Resume Screening AI
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .feature_engineering import load_vectorizer

def calculate_similarity(job_description, resume_text, vectorizer):
    """
    Calculate cosine similarity between job description and resume
    
    Args:
        job_description (str): Job description text
        resume_text (str): Resume text
        vectorizer: TF-IDF vectorizer
        
    Returns:
        float: Similarity score (0-1)
    """
    # Transform texts using the same vectorizer
    job_vector = vectorizer.transform([job_description])
    resume_vector = vectorizer.transform([resume_text])
    
    # Calculate cosine similarity
    similarity = cosine_similarity(job_vector, resume_vector)[0][0]
    
    return similarity

def rank_resumes(job_description, resumes_df, vectorizer, top_k=5):
    """
    Rank resumes based on similarity to job description
    
    Args:
        job_description (str): Job description text
        resumes_df (pd.DataFrame): DataFrame with resumes
        vectorizer: TF-IDF vectorizer
        top_k (int): Number of top resumes to return
        
    Returns:
        list: List of ranked resumes with scores
    """
    # Calculate similarity scores
    similarities = []
    
    for idx, row in resumes_df.iterrows():
        resume_text = row['cleaned_text'] if 'cleaned_text' in row else row['resume_text']
        score = calculate_similarity(job_description, resume_text, vectorizer)
        
        similarities.append({
            'resume_id': idx,
            'category': row['category'],
            'similarity_score': score,
            'resume_text': row['resume_text'][:200] + '...' if len(row['resume_text']) > 200 else row['resume_text']
        })
    
    # Sort by similarity score (descending)
    similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    # Return top k results
    return similarities[:top_k]

def get_match_score(similarity_score):
    """
    Convert similarity score to match percentage
    
    Args:
        similarity_score (float): Cosine similarity score (0-1)
        
    Returns:
        float: Match percentage (0-100)
    """
    return similarity_score * 100

def explain_match(job_description, resume_text, vectorizer, top_features=5):
    """
    Explain why a resume matches a job description
    
    Args:
        job_description (str): Job description text
        resume_text (str): Resume text
        vectorizer: TF-IDF vectorizer
        top_features (int): Number of top features to return
        
    Returns:
        dict: Explanation with top matching keywords
    """
    # Get feature names from vectorizer
    feature_names = vectorizer.get_feature_names_out()
    
    # Transform texts
    job_vector = vectorizer.transform([job_description])
    resume_vector = vectorizer.transform([resume_text])
    
    # Get feature importance
    job_scores = job_vector.toarray()[0]
    resume_scores = resume_vector.toarray()[0]
    
    # Find common important features
    common_features = []
    for i, (job_score, resume_score) in enumerate(zip(job_scores, resume_scores)):
        if job_score > 0 and resume_score > 0:
            common_features.append({
                'feature': feature_names[i],
                'job_score': job_score,
                'resume_score': resume_score,
                'combined_score': job_score * resume_score
            })
    
    # Sort by combined score
    common_features.sort(key=lambda x: x['combined_score'], reverse=True)
    
    return {
        'top_keywords': common_features[:top_features],
        'total_common_features': len(common_features)
    }

if __name__ == "__main__":
    # Test the ranking system
    import sys
    sys.path.append('..')
    from load_dataset import load_dataset
    from preprocessing import preprocess_dataset
    
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
