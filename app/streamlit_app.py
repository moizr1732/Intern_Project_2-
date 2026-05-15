#!/usr/bin/env python3
"""
Streamlit UI for Resume Screening AI
Dark AI Dashboard Theme
"""

import streamlit as st
import sys
import os
sys.path.append('..')

# Import modules
from utils.preprocessing import clean_text
from utils.feature_engineering import load_vectorizer
from utils.training import load_model
from utils.ranking import calculate_similarity, get_match_score, explain_match

# Dark AI Dashboard Theme CSS
st.markdown("""
<style>
    /* Main application background */
    .stApp {
        background: #0E1117;
        color: #FFFFFF;
    }
    
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
        background: #0E1117;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: #1C1F26;
        border-right: 1px solid #2A2D35;
    }
    
    .css-1d391kg .stSelectbox > div > div {
        background: #2A2D35;
        color: #FFFFFF;
        border: 1px solid #4F8CFF;
        border-radius: 8px;
    }
    
    .css-1d391kg .stSelectbox label {
        color: #A0A0A0;
        font-weight: 500;
    }
    
    /* Header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #FFFFFF;
        text-align: center;
        margin-bottom: 1rem;
        margin-top: 1rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #A0A0A0;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Card styling */
    .metric-card {
        background: #1C1F26;
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid #2A2D35;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        transition: all 0.2s ease;
        margin: 1rem 0;
    }
    
    .metric-card:hover {
        border-color: #4F8CFF;
        box-shadow: 0 6px 20px rgba(79, 140, 255, 0.1);
    }
    
    /* Success message styling */
    .success-message {
        background: #00C9A7;
        color: #FFFFFF;
        padding: 1rem;
        border-radius: 8px;
        border: none;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(0, 201, 167, 0.2);
    }
    
    /* Error message styling */
    .error-message {
        background: #FF6B6B;
        color: #FFFFFF;
        padding: 1rem;
        border-radius: 8px;
        border: none;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(255, 107, 107, 0.2);
    }
    
    /* Warning message styling */
    .warning-message {
        background: #FFA500;
        color: #FFFFFF;
        padding: 1rem;
        border-radius: 8px;
        border: none;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(255, 165, 0, 0.2);
    }
    
    /* Primary button styling */
    .stButton > button {
        background: #4F8CFF;
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(79, 140, 255, 0.2);
    }
    
    .stButton > button:hover {
        background: #3D7AE8;
        box-shadow: 0 4px 12px rgba(79, 140, 255, 0.3);
        transform: translateY(-1px);
    }
    
    /* Input field styling */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stFileUploader > div > div {
        background: #1C1F26;
        border: 1px solid #2A2D35;
        border-radius: 8px;
        color: #FFFFFF;
        font-size: 0.95rem;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #4F8CFF;
        box-shadow: 0 0 0 2px rgba(79, 140, 255, 0.1);
        outline: none;
    }
    
    .stTextInput > div > div > input::placeholder,
    .stTextArea > div > div > textarea::placeholder {
        color: #A0A0A0;
    }
    
    /* Header text improvements */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF;
        font-weight: 600;
    }
    
    /* Text improvements */
    p, div, span, label {
        color: #A0A0A0;
    }
    
    /* Strong/bold text */
    strong, b {
        color: #FFFFFF;
    }
</style>
""", unsafe_allow_html=True)

# Page configuration
st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main application function"""
    # Title and subtitle
    st.markdown('<h1 class="main-header">📄 AI Resume Screening System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Smarter hiring with Machine Learning</p>', unsafe_allow_html=True)
    
    # Load models and vectorizer
    model = load_model()
    vectorizer = load_vectorizer()
    
    if model is None or vectorizer is None:
        st.markdown("""
        <div class="error-message">
            ⚠️ Models not found! Please run <code>python main.py</code> first to train the models.
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Main input interface
    st.markdown("---")
    
    # Resume Input Section
    st.subheader("📄 Resume Input")
    
    resume_text = st.text_area(
        "Enter resume text:",
        height=150,
        placeholder="Enter the complete resume text here for analysis...",
        key="resume_input"
    )
    
    # Job Description Section
    st.subheader("💼 Job Description")
    job_description = st.text_area(
        "Enter job description:",
        height=100,
        placeholder="e.g., Looking for a Data Scientist with Python, machine learning, and SQL experience...",
        key="job_desc_input"
    )
    
    # Analyze Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🔍 Analyze Resume", type="primary", use_container_width=True, key="analyze")
    
    # Results Section
    if analyze_button and resume_text:
        st.markdown("---")
        st.subheader("📊 Analysis Results")
        
        with st.spinner("Analyzing resume..."):
            try:
                # Preprocess resume text
                cleaned_resume = clean_text(resume_text)
                
                # Predict category
                resume_vector = vectorizer.transform([cleaned_resume])
                prediction = model.predict(resume_vector)[0]
                
                # Get confidence score
                if hasattr(model, 'predict_proba'):
                    probabilities = model.predict_proba(resume_vector)[0]
                    confidence = max(probabilities)
                else:
                    confidence = 0.8  # Default confidence
                
                # Calculate similarity if job description provided
                match_score = 0.0
                explanation_text = ""
                top_keywords = []
                
                if job_description:
                    similarity = calculate_similarity(job_description, cleaned_resume, vectorizer)
                    match_score = get_match_score(similarity)
                    
                    # Get explanation
                    explanation = explain_match(job_description, cleaned_resume, vectorizer)
                    top_keywords = [k['feature'] for k in explanation['top_keywords'][:5]]
                    
                    if len(top_keywords) > 0:
                        explanation_text = f"Found {explanation['total_common_features']} matching keywords"
                
                # Display results in cards
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #4F8CFF; margin-bottom: 0.5rem;">🎯 Predicted Role</h4>
                        <p style="color: #FFFFFF; font-size: 1.2rem; font-weight: 600;">{prediction}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #00C9A7; margin-bottom: 0.5rem;">📈 Confidence</h4>
                        <p style="color: #FFFFFF; font-size: 1.2rem; font-weight: 600;">{confidence:.1%}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    word_count = len(resume_text.split())
                    st.markdown(f"""
                    <div class="metric-card">
                        <h4 style="color: #FFA500; margin-bottom: 0.5rem;">📝 Word Count</h4>
                        <p style="color: #FFFFFF; font-size: 1.2rem; font-weight: 600;">{word_count}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Job Match Score
                if job_description:
                    st.subheader("🏆 Job Match Analysis")
                    
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="color: #4F8CFF; margin-bottom: 0.5rem;">🎯 Job Match Score</h4>
                            <p style="color: #FFFFFF; font-size: 1.2rem; font-weight: 600;">{match_score:.1f}%</p>
                            <p style="color: #A0A0A0; font-size: 0.9rem;">How well this resume matches the job description</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        if match_score >= 70:
                            score_color = "#00C9A7"
                            score_text = "Excellent Match"
                        elif match_score >= 50:
                            score_color = "#FFA500"
                            score_text = "Good Match"
                        else:
                            score_color = "#FF6B6B"
                            score_text = "Poor Match"
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <h4 style="color: {score_color}; margin-bottom: 0.5rem;">📊 Match Level</h4>
                            <p style="color: #FFFFFF; font-size: 1.0rem; font-weight: 600;">{score_text}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Explanation section
                    if explanation_text and top_keywords:
                        st.subheader("🔍 Explanation")
                        
                        st.markdown(f"""
                        <div class="metric-card">
                            <p style="color: #A0A0A0; line-height: 1.6;">{explanation_text}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.write("**Top Matching Keywords:**")
                        keywords_text = ", ".join([f"`{keyword}`" for keyword in top_keywords])
                        st.markdown(f"<div style='color: #4F8CFF; font-size: 1.0rem;'>{keywords_text}</div>", unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f"""
                <div class="error-message">
                    ❌ Error during analysis: {str(e)}
                </div>
                """, unsafe_allow_html=True)
    
    # Instructions section
    st.markdown("---")
    st.subheader("📖 How to Use")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #4F8CFF; margin-bottom: 1rem;">📝 Step 1: Input Resume</h4>
            <p style="color: #A0A0A0;">Paste the complete resume text in the text area above.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4 style="color: #4F8CFF; margin-bottom: 1rem;">💼 Step 2: Add Job Description</h4>
            <p style="color: #A0A0A0;">Optional: Add a job description for match scoring.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="metric-card">
        <h4 style="color: #00C9A7; margin-bottom: 1rem;">🔍 Step 3: Analyze</h4>
        <p style="color: #A0A0A0;">Click "Analyze Resume" to get instant results including predicted role, confidence score, and job match percentage.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
