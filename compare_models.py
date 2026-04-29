"""
Model Comparison Script
=======================
Compares different vectorizers and classifiers for spam detection.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

from spam_detector import SpamDetector


def compare_models(filepath):
    """
    Compare all combinations of vectorizers and models.
    
    Args:
        filepath: Path to the dataset
    """
    print("="*60)
    print("MODEL COMPARISON: SPAM DETECTOR")
    print("="*60)
    
    # Load and prepare data using one detector instance
    base_detector = SpamDetector()
    df = base_detector.load_data(filepath)
    
    # Preprocess text
    import re
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    
    stemmer = PorterStemmer()
    try:
        stop_words = set(stopwords.words('english'))
    except:
        stop_words = set()
    
    def preprocess(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = text.split()
        words = [stemmer.stem(w) for w in words if w not in stop_words]
        return ' '.join(words)
    
    df['processed'] = df['message'].apply(preprocess)
    df['label_binary'] = df['label'].map({'spam': 1, 'ham': 0})
    
    X = df['processed']
    y = df['label_binary']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define configurations to compare
    configs = [
        ('Bag of Words', CountVectorizer(max_features=3000), 'Naive Bayes', MultinomialNB()),
        ('TF-IDF', TfidfVectorizer(max_features=3000), 'Naive Bayes', MultinomialNB()),
        ('Bag of Words', CountVectorizer(max_features=3000), 'Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
        ('TF-IDF', TfidfVectorizer(max_features=3000), 'Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
    ]
    
    results = []
    
    for vec_name, vectorizer, model_name, model in configs:
        # Vectorize
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train
        model.fit(X_train_vec, y_train)
        
        # Predict
        preds = model.predict(X_test_vec)
        
        # Metrics
        results.append({
            'Vectorizer': vec_name,
            'Model': model_name,
            'Accuracy': accuracy_score(y_test, preds),
            'Precision': precision_score(y_test, preds),
            'Recall': recall_score(y_test, preds),
            'F1 Score': f1_score(y_test, preds)
        })
    
    # Display results as a table
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('F1 Score', ascending=False)
    
    print("\nResults (sorted by F1 Score):\n")
    print(results_df.to_string(index=False, float_format='{:.4f}'.format))
    
    # Highlight best model
    best = results_df.iloc[0]
    print(f"\nBest Model: {best['Vectorizer']} + {best['Model']}")
    print(f"  Accuracy:  {best['Accuracy']:.4f}")
    print(f"  Precision: {best['Precision']:.4f}")
    print(f"  Recall:    {best['Recall']:.4f}")
    print(f"  F1 Score:  {best['F1 Score']:.4f}")
    
    return results_df


if __name__ == "__main__":
    try:
        compare_models('spam.csv')
    except FileNotFoundError:
        print("Error: 'spam.csv' not found. Please download the dataset first.")
        print("Run: python download_dataset.py")
