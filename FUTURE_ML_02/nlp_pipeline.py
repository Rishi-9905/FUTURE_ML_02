import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def clean_text(text):
    # Lowercase
    text = str(text).lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    # 1. Load Data
    try:
        df = pd.read_csv('data/synthetic_tickets.csv')
    except FileNotFoundError:
        print("Data file not found. Run ticket_data_generator.py first.")
        return

    # 2. Text Preprocessing
    print("Cleaning text data...")
    df['Clean_Text'] = df['Text'].apply(clean_text)

    # 3. Feature Extraction (TF-IDF)
    print("Extracting TF-IDF features...")
    tfidf = TfidfVectorizer(stop_words='english', max_features=1000)
    X = tfidf.fit_transform(df['Clean_Text'])
    
    # Define targets
    y_category = df['Category']
    y_priority = df['Priority']

    # Split data
    X_train, X_test, y_cat_train, y_cat_test, y_prio_train, y_prio_test = train_test_split(
        X, y_category, y_priority, test_size=0.2, random_state=42
    )

    # 4. Train Models
    print("Training Category Classifier...")
    cat_model = RandomForestClassifier(n_estimators=100, random_state=42)
    cat_model.fit(X_train, y_cat_train)

    print("Training Priority Classifier...")
    prio_model = RandomForestClassifier(n_estimators=100, random_state=42)
    prio_model.fit(X_train, y_prio_train)

    # 5. Evaluate
    print("\n--- Category Classification Results ---")
    cat_preds = cat_model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_cat_test, cat_preds):.4f}")
    print(classification_report(y_cat_test, cat_preds))

    print("\n--- Priority Classification Results ---")
    prio_preds = prio_model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_prio_test, prio_preds):.4f}")
    print(classification_report(y_prio_test, prio_preds))

    # 6. Visualizations
    plt.figure(figsize=(8, 6))
    cm_cat = confusion_matrix(y_cat_test, cat_preds, labels=cat_model.classes_)
    sns.heatmap(cm_cat, annot=True, fmt='d', cmap='Blues', 
                xticklabels=cat_model.classes_, yticklabels=cat_model.classes_)
    plt.title('Confusion Matrix: Ticket Category')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('category_confusion_matrix.png', dpi=300)
    print("Saved category matrix to category_confusion_matrix.png")

    plt.figure(figsize=(8, 6))
    cm_prio = confusion_matrix(y_prio_test, prio_preds, labels=prio_model.classes_)
    sns.heatmap(cm_prio, annot=True, fmt='d', cmap='Reds',
                xticklabels=prio_model.classes_, yticklabels=prio_model.classes_)
    plt.title('Confusion Matrix: Ticket Priority')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('priority_confusion_matrix.png', dpi=300)
    print("Saved priority matrix to priority_confusion_matrix.png")

if __name__ == '__main__':
    main()
