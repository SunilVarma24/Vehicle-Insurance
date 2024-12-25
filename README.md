![image](https://github.com/user-attachments/assets/532562a5-698e-41da-b75c-736d584acb1f)

# Vehicle Insurance Prediction using Machine Learning

## Project Overview
This project predicts whether past health insurance customers will be interested in vehicle insurance using demographic, vehicle, and policy data. The dataset is highly imbalanced, and techniques like ENN and SMOTETomek were applied to balance it. An XGBoost model was used, achieving 93% accuracy.

## Introduction
Insurance companies benefit from machine learning to improve customer engagement and optimize their services. This project focuses on predicting vehicle insurance interest among past health insurance policyholders. Accurate predictions enable the company to target customers effectively, increasing revenue and operational efficiency.

## Dataset Description
The dataset contains:
- **Demographics**: Gender, age, region code type  
- **Vehicles**: Vehicle age, history of damage  
- **Policy Details**: Premium, sourcing channel  

The target variable is highly imbalanced:
- **1**: Interested in vehicle insurance  
- **0**: Not interested in vehicle insurance  

## Methodology
1. **Data Preprocessing**:
   - Removed noisy and overlapping samples using Edited Nearest Neighbors (ENN).
   - Applied SMOTETomek for under- and oversampling to balance the data and create synthetic samples.

2. **Feature Engineering**:
   - One-hot encoding for categorical variables.
   - Normalized numerical variables.

3. **Model Training**:
   - Used the XGBoost algorithm due to its robustness and efficiency with structured data.
   - Evaluated the model on validation and test sets.

4. **Performance Metrics**:
   - Accuracy: 93%
   - Precision, Recall, and F1-Score were also analyzed to ensure balance.

## Results
The model effectively predicts vehicle insurance interest with consistent performance:
- **Accuracy**: 93% on both validation and test datasets.
- **Balanced Dataset**: Improved prediction for the minority class after applying ENN and SMOTETomek techniques.

## Installation
To replicate this project, install the necessary libraries:
```bash
pip install numpy pandas scikit-learn xgboost imbalanced-learn matplotlib seaborn
