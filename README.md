# FUTURE_ML_02
🎫 An NLP Machine Learning pipeline that automates customer support ticket triage by predicting issue Category and Urgency (Priority) using TF-IDF and Random Forests.
# 🎫 Support Ticket Classification & Prioritization

## Overview
This project is an NLP-based Machine Learning pipeline built for **Future Interns Task 2**. It automates the triage of customer support tickets by reading unstructured text, categorizing the issue, and mathematically predicting the required priority level (High/Medium/Low) to enforce SLAs and reduce response times.

## Features
- **Data Generation**: `ticket_data_generator.py` intelligently synthesizes a dataset containing 2,000 uniquely structured text-based customer complaints and queries.
- **NLP Pipeline**: `nlp_pipeline.py` cleans text (removes stopwords, punctuation), extracts computational features using `TfidfVectorizer`, and trains dual `RandomForestClassifier` algorithms.
- **Business Insights**: Automatically evaluates classification metrics and plots the model's predictive ability through clean Seaborn confusion matrix heatmaps (`category_confusion_matrix.png` and `priority_confusion_matrix.png`).

## Usage
1. Run `python ticket_data_generator.py` to create the mock NLP dataset (`synthetic_tickets.csv`) locally inside the `data/` folder.
2. Run `python nlp_pipeline.py` to clean the text, build the TF-IDF vocabulary, train both classifiers, and generate the final model evaluations and PNG plots.
