import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from typing import List, Dict, Tuple, Any, Optional
import re
from collections import Counter
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ExplainableAI:
    """
    A comprehensive explainable AI system for resume classification insights.
    """
    
    def __init__(self, model=None, feature_engineering=None):
        """
        Initialize the explainable AI system.
        
        Args:
            model: Trained classification model
            feature_engineering: Fitted feature engineering instance
        """
        self.model = model
        self.feature_engineering = feature_engineering
        self.explanations = {}
        
        logger.info("ExplainableAI initialized")
    
    def get_prediction_explanation(self, resume_text: str, job_description: str = None) -> Dict[str, Any]:
        """
        Get comprehensive explanation for a resume prediction.
        
        Args:
            resume_text (str): Resume text to analyze
            job_description (str): Optional job description for context
            
        Returns:
            Dict[str, Any]: Detailed explanation
        """
        try:
            if not self.model or not self.feature_engineering:
                raise ValueError("Model and feature engineering must be provided")
            
            # Preprocess text
            from preprocessing import TextPreprocessor
            preprocessor = TextPreprocessor()
            processed_text = preprocessor.preprocess_text(resume_text)
            
            # Get prediction
            features = self.feature_engineering.extract_tfidf_features([processed_text])
            prediction = self.model.predict(features)[0]
            probabilities = None
            
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features)[0]
                confidence = np.max(probabilities)
            else:
                confidence = 1.0
            
            # Get top keywords
            keywords = self.feature_engineering.get_top_keywords(processed_text, top_k=20)
            
            # Get feature importance if available
            feature_importance = self._get_feature_importance_for_text(features[0])
            
            # Generate explanation
            explanation = {
                'predicted_category': prediction,
                'confidence_score': confidence,
                'probabilities': {f'Class_{i}': prob for i, prob in enumerate(probabilities)} if probabilities is not None else None,
                'top_keywords': keywords,
                'feature_importance': feature_importance,
                'explanation_text': self._generate_prediction_explanation(
                    prediction, keywords, confidence, job_description
                ),
                'text_statistics': self._get_text_statistics(resume_text)
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating prediction explanation: {e}")
            return {'error': str(e)}
    
    def _get_feature_importance_for_text(self, feature_vector: np.ndarray) -> Dict[str, float]:
        """
        Get feature importance for a specific text.
        
        Args:
            feature_vector (np.ndarray): TF-IDF feature vector
            
        Returns:
            Dict[str, float]: Feature importance dictionary
        """
        try:
            if not hasattr(self.model, 'feature_importances_'):
                return {}
            
            # Get feature names
            feature_names = self.feature_engineering.feature_names
            
            # Get non-zero features
            non_zero_indices = np.where(feature_vector > 0)[0]
            
            # Calculate importance for active features
            importance_dict = {}
            for idx in non_zero_indices:
                if idx < len(feature_names):
                    feature_name = feature_names[idx]
                    feature_value = feature_vector[idx]
                    model_importance = self.model.feature_importances_[idx]
                    
                    # Combined importance (TF-IDF score * model importance)
                    combined_importance = feature_value * model_importance
                    importance_dict[feature_name] = combined_importance
            
            # Sort by importance
            importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
            return importance_dict
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}
    
    def _generate_prediction_explanation(self, prediction: str, keywords: List[Tuple[str, float]], 
                                       confidence: float, job_description: str = None) -> str:
        """
        Generate human-readable explanation for prediction.
        
        Args:
            prediction (str): Predicted category
            keywords (List[Tuple[str, float]]): Top keywords with scores
            confidence (float): Confidence score
            job_description (str): Optional job description
            
        Returns:
            str: Explanation text
        """
        try:
            # Get top keywords
            top_keywords = [word for word, score in keywords[:10]]
            
            # Base explanation
            explanation = (f"This resume is classified as **{prediction}** with a confidence score of "
                          f"**{confidence:.3f}**. ")
            
            # Add keywords explanation
            if top_keywords:
                keywords_str = ', '.join(top_keywords[:5])
                explanation += f"The classification is primarily based on key skills and qualifications: **{keywords_str}**. "
                
                # Add more detail about keywords
                if len(top_keywords) > 5:
                    additional_keywords = ', '.join(top_keywords[5:8])
                    explanation += f"Additional relevant terms include: {additional_keywords}. "
            
            # Add job description context if provided
            if job_description:
                job_keywords = self.feature_engineering.get_top_keywords(job_description, top_k=10)
                job_keyword_words = [word for word, score in job_keywords]
                common_keywords = list(set(top_keywords) & set(job_keyword_words))
                
                if common_keywords:
                    common_str = ', '.join(common_keywords[:3])
                    explanation += f"The resume shares **{len(common_keywords)}** key terms with the job description: {common_str}. "
                else:
                    explanation += "The resume shows limited overlap with the job description keywords. "
            
            # Add confidence interpretation
            if confidence > 0.8:
                explanation += "The model is **highly confident** in this classification."
            elif confidence > 0.6:
                explanation += "The model is **moderately confident** in this classification."
            else:
                explanation += "The model has **low confidence** in this classification and recommends human review."
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation text: {e}")
            return f"Resume classified as {prediction} with confidence {confidence:.3f}."
    
    def _get_text_statistics(self, text: str) -> Dict[str, Any]:
        """
        Get statistics about the input text.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, Any]: Text statistics
        """
        try:
            if not isinstance(text, str):
                return {}
            
            words = text.split()
            sentences = text.split('.')
            
            stats = {
                'character_count': len(text),
                'word_count': len(words),
                'sentence_count': len([s for s in sentences if s.strip()]),
                'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
                'unique_words': len(set(words)),
                'lexical_diversity': len(set(words)) / len(words) if words else 0
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating text statistics: {e}")
            return {}
    
    def get_keyword_analysis(self, category: str, top_k: int = 20) -> Dict[str, Any]:
        """
        Get keyword analysis for a specific category.
        
        Args:
            category (str): Category to analyze
            top_k (int): Number of top keywords to return
            
        Returns:
            Dict[str, Any]: Keyword analysis results
        """
        try:
            if not hasattr(self, 'category_data'):
                logger.warning("No category data available for analysis")
                return {}
            
            if category not in self.category_data:
                logger.warning(f"Category {category} not found")
                return {}
            
            category_texts = self.category_data[category]
            
            # Combine all texts for the category
            combined_text = ' '.join(category_texts)
            
            # Get keywords
            keywords = self.feature_engineering.get_top_keywords(combined_text, top_k)
            
            # Calculate keyword statistics
            keyword_freq = Counter()
            for text in category_texts:
                words = text.lower().split()
                for word, _ in keywords:
                    if word in words:
                        keyword_freq[word] += 1
            
            # Create analysis
            analysis = {
                'category': category,
                'total_documents': len(category_texts),
                'top_keywords': keywords,
                'keyword_frequency': dict(keyword_freq.most_common(top_k)),
                'keyword_coverage': {word: freq/len(category_texts) for word, freq in keyword_freq.items()},
                'analysis_text': self._generate_keyword_analysis_text(category, keywords, keyword_freq)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in keyword analysis: {e}")
            return {}
    
    def _generate_keyword_analysis_text(self, category: str, keywords: List[Tuple[str, float]], 
                                      keyword_freq: Counter) -> str:
        """
        Generate text analysis for keywords.
        
        Args:
            category (str): Category name
            keywords (List[Tuple[str, float]]): Top keywords
            keyword_freq (Counter): Keyword frequency counter
            
        Returns:
            str: Analysis text
        """
        try:
            top_keywords = [word for word, score in keywords[:10]]
            
            analysis = (f"The **{category}** category is characterized by key terms such as: "
                       f"**{', '.join(top_keywords[:5])}**. ")
            
            # Add frequency information
            if keyword_freq:
                most_common = keyword_freq.most_common(1)[0]
                analysis += f"The most frequent term is '{most_common[0]}' appearing in {most_common[1]} documents. "
            
            # Add coverage information
            high_coverage = [word for word, freq in keyword_freq.items() if freq > len(keyword_freq) * 0.5]
            if high_coverage:
                analysis += f"Terms like {', '.join(high_coverage[:3])} appear in more than 50% of documents. "
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating keyword analysis text: {e}")
            return f"Keyword analysis for {category} completed."
    
    def plot_keyword_importance(self, keywords: List[Tuple[str, float]], title: str = "Keyword Importance",
                              save_path: str = None) -> plt.Figure:
        """
        Plot keyword importance chart.
        
        Args:
            keywords (List[Tuple[str, float]]): Keywords with importance scores
            title (str): Plot title
            save_path (str): Path to save the plot
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        try:
            if not keywords:
                logger.warning("No keywords provided for plotting")
                return None
            
            # Prepare data
            words = [word for word, score in keywords[:15]]
            scores = [score for word, score in keywords[:15]]
            
            # Create plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Create horizontal bar plot
            bars = ax.barh(words, scores, color=sns.color_palette("viridis", len(words)))
            
            # Customize plot
            ax.set_title(title, fontsize=16, fontweight='bold')
            ax.set_xlabel('Importance Score', fontsize=12)
            ax.set_ylabel('Keywords', fontsize=12)
            
            # Add value labels
            for i, (bar, score) in enumerate(zip(bars, scores)):
                ax.text(score + max(scores) * 0.01, bar.get_y() + bar.get_height()/2,
                       f'{score:.3f}', ha='left', va='center', fontsize=10)
            
            # Invert y-axis to show highest importance at top
            ax.invert_yaxis()
            
            # Add grid
            ax.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Keyword importance plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting keyword importance: {e}")
            return None
    
    def plot_prediction_confidence(self, probabilities: np.ndarray, class_names: List[str],
                                 save_path: str = None) -> plt.Figure:
        """
        Plot prediction confidence chart.
        
        Args:
            probabilities (np.ndarray): Prediction probabilities
            class_names (List[str]): Class names
            save_path (str): Path to save the plot
            
        Returns:
            plt.Figure: Matplotlib figure
        """
        try:
            if probabilities is None or len(probabilities) == 0:
                logger.warning("No probabilities provided for plotting")
                return None
            
            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create bar plot
            colors = sns.color_palette("husl", len(probabilities))
            bars = ax.bar(class_names, probabilities, color=colors)
            
            # Customize plot
            ax.set_title('Prediction Confidence by Category', fontsize=16, fontweight='bold')
            ax.set_ylabel('Confidence Score', fontsize=12)
            ax.set_xlabel('Category', fontsize=12)
            ax.set_ylim(0, 1)
            
            # Add value labels
            for bar, prob in zip(bars, probabilities):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{prob:.3f}', ha='center', va='bottom', fontsize=10)
            
            # Rotate x-axis labels if needed
            if len(max(class_names, key=len)) > 10:
                plt.xticks(rotation=45, ha='right')
            
            # Add grid
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Prediction confidence plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting prediction confidence: {e}")
            return None
    
    def generate_explainability_report(self, resume_text: str, job_description: str = None,
                                    save_path: str = None) -> str:
        """
        Generate comprehensive explainability report.
        
        Args:
            resume_text (str): Resume text to analyze
            job_description (str): Optional job description
            save_path (str): Path to save the report
            
        Returns:
            str: Explainability report
        """
        try:
            # Get prediction explanation
            explanation = self.get_prediction_explanation(resume_text, job_description)
            
            if 'error' in explanation:
                return f"Error generating report: {explanation['error']}"
            
            # Generate report
            report = "# Resume Classification Explainability Report\n\n"
            
            # Prediction summary
            report += "## Prediction Summary\n\n"
            report += f"- **Predicted Category**: {explanation['predicted_category']}\n"
            report += f"- **Confidence Score**: {explanation['confidence_score']:.3f}\n"
            
            if explanation['probabilities']:
                report += "\n### Prediction Probabilities\n\n"
                for class_name, prob in explanation['probabilities'].items():
                    report += f"- **{class_name}**: {prob:.3f}\n"
            
            # Explanation
            report += "\n## Explanation\n\n"
            report += explanation['explanation_text'] + "\n\n"
            
            # Top keywords
            report += "## Top Keywords\n\n"
            report += "| Keyword | Score |\n"
            report += "|---------|-------|\n"
            for word, score in explanation['top_keywords'][:15]:
                report += f"| {word} | {score:.4f} |\n"
            
            # Feature importance
            if explanation['feature_importance']:
                report += "\n## Feature Importance\n\n"
                report += "| Feature | Importance |\n"
                report += "|---------|------------|\n"
                for feature, importance in list(explanation['feature_importance'].items())[:10]:
                    report += f"| {feature} | {importance:.4f} |\n"
            
            # Text statistics
            if explanation['text_statistics']:
                report += "\n## Text Statistics\n\n"
                stats = explanation['text_statistics']
                report += f"- **Character Count**: {stats.get('character_count', 0)}\n"
                report += f"- **Word Count**: {stats.get('word_count', 0)}\n"
                report += f"- **Sentence Count**: {stats.get('sentence_count', 0)}\n"
                report += f"- **Average Word Length**: {stats.get('avg_word_length', 0):.2f}\n"
                report += f"- **Unique Words**: {stats.get('unique_words', 0)}\n"
                report += f"- **Lexical Diversity**: {stats.get('lexical_diversity', 0):.3f}\n"
            
            # Save report if requested
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'w') as f:
                    f.write(report)
                logger.info(f"Explainability report saved to {save_path}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating explainability report: {e}")
            return f"Error generating report: {e}"

def create_explainable_ai(model, feature_engineering, resume_dataset: pd.DataFrame = None) -> ExplainableAI:
    """
    Create and initialize an explainable AI system.
    
    Args:
        model: Trained classification model
        feature_engineering: Fitted feature engineering instance
        resume_dataset (pd.DataFrame): Resume dataset for category analysis
        
    Returns:
        ExplainableAI: Initialized explainable AI system
    """
    try:
        xai = ExplainableAI(model, feature_engineering)
        
        # Add category data if provided
        if resume_dataset is not None and 'category' in resume_dataset.columns and 'resume_text_processed' in resume_dataset.columns:
            category_data = {}
            for category in resume_dataset['category'].unique():
                category_texts = resume_dataset[resume_dataset['category'] == category]['resume_text_processed'].tolist()
                category_data[category] = category_texts
            
            xai.category_data = category_data
            logger.info(f"Category data loaded for {len(category_data)} categories")
        
        logger.info("ExplainableAI system created and initialized")
        return xai
        
    except Exception as e:
        logger.error(f"Error creating explainable AI: {e}")
        raise

if __name__ == "__main__":
    try:
        # Test the explainable AI system
        from training import ModelTrainer
        from preprocessing import load_and_preprocess_resume_data
        from feature_engineering import create_features_pipeline
        
        # Load and preprocess data
        df_processed, metadata = load_and_preprocess_resume_data()
        
        # Create features and train model
        feature_results = create_features_pipeline(df_processed)
        X_train = feature_results['X_train']
        X_test = feature_results['X_test']
        y_train = feature_results['y_train']
        y_test = feature_results['y_test']
        
        # Train model
        trainer = ModelTrainer()
        training_results = trainer.train_all_models(X_train, y_train, X_test, y_test)
        
        # Create explainable AI
        xai = create_explainable_ai(trainer.best_model, feature_results['feature_engineering'], df_processed)
        
        # Test explanation
        sample_text = df_processed['resume_text'].iloc[0]
        job_desc = "Looking for a data scientist with Python and machine learning experience"
        
        explanation = xai.get_prediction_explanation(sample_text, job_desc)
        
        print("=== Explainable AI Results ===")
        print(f"Predicted: {explanation.get('predicted_category', 'Unknown')}")
        print(f"Confidence: {explanation.get('confidence_score', 0):.3f}")
        print(f"Top keywords: {[word for word, score in explanation.get('top_keywords', [])[:5]]}")
        
    except Exception as e:
        print(f"Error in explainable AI test: {e}")
