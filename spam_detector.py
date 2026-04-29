"""
Spam Mail Detector
==================
A text classification project that distinguishes between spam and ham (non-spam) emails.
Uses the SMS Spam Collection dataset and implements various ML techniques.
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
print("Downloading NLTK data...")
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

class SpamDetector:
    """
    A spam detection classifier with text preprocessing and multiple model options.
    """
    
    def __init__(self, vectorizer_type='tfidf', model_type='naive_bayes'):
        """
        Initialize the spam detector.
        
        Args:
            vectorizer_type: 'bow' for Bag of Words or 'tfidf' for TF-IDF
            model_type: 'naive_bayes' or 'logistic_regression'
        """
        self.vectorizer_type = vectorizer_type
        self.model_type = model_type
        self.stemmer = PorterStemmer()
        
        # Initialize vectorizer
        if vectorizer_type == 'bow':
            self.vectorizer = CountVectorizer(max_features=3000)
        else:
            self.vectorizer = TfidfVectorizer(max_features=3000)
        
        # Initialize model
        if model_type == 'naive_bayes':
            self.model = MultinomialNB()
        else:
            self.model = LogisticRegression(max_iter=1000, random_state=42)
        
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
    
    def preprocess_text(self, text):
        """
        Preprocess text data: lowercase, remove special chars, remove stopwords, stem.
        
        Args:
            text: Input text string
            
        Returns:
            Preprocessed text string
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenization
        words = text.split()
        
        # Remove stopwords and apply stemming
        words = [self.stemmer.stem(word) for word in words if word not in self.stop_words]
        
        return ' '.join(words)
    
    def load_data(self, filepath):
        """
        Load the SMS Spam Collection dataset.
        
        Args:
            filepath: Path to the dataset file
            
        Returns:
            DataFrame with messages and labels
        """
        # Load dataset (tab-separated)
        df = pd.read_csv(filepath, sep='\t', names=['label', 'message'], encoding='latin-1')
        
        print(f"Dataset loaded: {len(df)} messages")
        print(f"Spam messages: {sum(df['label'] == 'spam')}")
        print(f"Ham messages: {sum(df['label'] == 'ham')}")
        
        return df
    
    def prepare_data(self, df):
        """
        Preprocess all messages and prepare features.
        
        Args:
            df: DataFrame with 'message' and 'label' columns
            
        Returns:
            X (features), y (labels)
        """
        print("\nPreprocessing text data...")
        
        # Preprocess all messages
        df['processed_message'] = df['message'].apply(self.preprocess_text)
        
        # Convert labels to binary (spam=1, ham=0)
        df['label_binary'] = df['label'].map({'spam': 1, 'ham': 0})
        
        return df['processed_message'], df['label_binary']
    
    def train(self, X_train, y_train):
        """
        Train the spam detection model.
        
        Args:
            X_train: Training messages
            y_train: Training labels
        """
        print(f"\nVectorizing text using {self.vectorizer_type.upper()}...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        
        print(f"Training {self.model_type.replace('_', ' ').title()} model...")
        self.model.fit(X_train_vec, y_train)
        
        print("Training complete!")
    
    def predict(self, X_test):
        """
        Make predictions on test data.
        
        Args:
            X_test: Test messages
            
        Returns:
            Predictions array
        """
        X_test_vec = self.vectorizer.transform(X_test)
        return self.model.predict(X_test_vec)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.
        
        Args:
            X_test: Test messages
            y_test: True labels
            
        Returns:
            Dictionary with performance metrics
        """
        predictions = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, predictions),
            'precision': precision_score(y_test, predictions),
            'recall': recall_score(y_test, predictions),
            'f1_score': f1_score(y_test, predictions)
        }
        
        print("\n" + "="*50)
        print("MODEL PERFORMANCE")
        print("="*50)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1_score']:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, predictions))
        print("\nClassification Report:")
        print(classification_report(y_test, predictions, target_names=['Ham', 'Spam']))
        
        return metrics
    
    def predict_message(self, message):
        """
        Predict if a single message is spam or ham.
        
        Args:
            message: Text message to classify
            
        Returns:
            Prediction ('spam' or 'ham') and probability
        """
        processed = self.preprocess_text(message)
        vectorized = self.vectorizer.transform([processed])
        prediction = self.model.predict(vectorized)[0]
        probability = self.model.predict_proba(vectorized)[0]
        
        label = 'spam' if prediction == 1 else 'ham'
        confidence = probability[prediction]
        
        return label, confidence


def main():
    """
    Main function to run the spam detector pipeline.
    """
    print("="*50)
    print("SPAM MAIL DETECTOR")
    print("="*50)
    
    # Initialize detector
    detector = SpamDetector(vectorizer_type='tfidf', model_type='naive_bayes')
    
    # Load data
    try:
        df = detector.load_data('spam.csv')
    except FileNotFoundError:
        print("\nError: 'spam.csv' not found!")
        print("Please download the SMS Spam Collection dataset from:")
        print("https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection")
        print("\nOr use the provided sample_data.csv for testing.")
        return
    
    # Prepare data
    X, y = detector.prepare_data(df)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {len(X_train)} messages")
    print(f"Test set:  {len(X_test)} messages")
    
    # Train model
    detector.train(X_train, y_train)
    
    # Evaluate model
    metrics = detector.evaluate(X_test, y_test)
    
    # Test with sample messages
    print("\n" + "="*50)
    print("TESTING WITH SAMPLE MESSAGES")
    print("="*50)
    
    sample_messages = [
        "Congratulations! You've won a $1000 gift card. Click here to claim now!",
        "Hey, are we still meeting for lunch tomorrow?",
        "URGENT: Your account will be suspended. Verify your identity immediately.",
        "Can you pick up some milk on your way home?",
        "FREE entry to win a brand new iPhone! Text WIN to 12345"
    ]
    
    for msg in sample_messages:
        label, confidence = detector.predict_message(msg)
        print(f"\nMessage: {msg[:60]}...")
        print(f"Prediction: {label.upper()} (confidence: {confidence:.2%})")


if __name__ == "__main__":
    main()
