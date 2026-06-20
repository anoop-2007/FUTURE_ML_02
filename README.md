# FUTURE_ML_02
# Support Ticket Classification & Prioritization

## About
An NLP-based machine learning system that automatically classifies customer
support tickets into categories and predicts their priority level, helping
support teams respond faster and reduce backlog.

## Problem
Support teams receive hundreds of tickets daily. Manually sorting and
prioritizing them wastes time and delays urgent issues. This project
automates both steps using text classification.

## Dataset
Customer Support Ticket Dataset (Kaggle)
https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset

## Approach
- Text cleaning (lowercasing, punctuation removal, stopword removal)
- Feature extraction using TF-IDF (unigrams + bigrams)
- Two separate classification models:
  - **Category prediction**: Billing, Technical Issue, Refund Request,
    Cancellation Request, Product Inquiry
  - **Priority prediction**: Low, Medium, High, Critical
- Model: Logistic Regression
- Evaluation: accuracy, precision, recall, F1-score, confusion matrices

## How Tickets Are Categorized
Ticket text is cleaned and converted into TF-IDF features, then a trained
Logistic Regression model predicts the most likely category based on
patterns learned from thousands of past labeled tickets.

## How Priority Is Decided
The same TF-IDF representation is fed into a separate classifier trained
to recognize urgency signals in ticket language, predicting Low/Medium/
High/Critical priority.

## Results
See `ticket_classifier.ipynb` for full evaluation metrics, confusion
matrices, and per-class performance breakdown.

## Tech Stack
Python · Pandas · NumPy · Scikit-learn · NLTK · Jupyter Notebook

## Files
- `ticket_classifier.ipynb` — full notebook: data loading, cleaning,
  preprocessing, model training, evaluation, and business explanation
