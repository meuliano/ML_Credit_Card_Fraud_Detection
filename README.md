<h1 align = center>Credit Card Fraud Detection Using Logistic Regression and Support Vector Machines</h1>

## Overview
Main code is FraudDection.ipynb

Old code used to develop this file is stored in the Development folder.

## Requirements:
- pandas
- numpy
- sklearn
- matplotlib

In this project Credit Card fraud detection was performed using Support Vector Machines and Logistic Regression models.
Data preprocessing and resampling was used to compensate for the imbalanced classes.

Database: [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle

## Data Visualzation
First, the dataset was loaded into the jupyter notebook using pandas.
The number of non fraudulent transactions are **284315** and the number of fradulent transactions are **492**.
Next, vsualization of cost and time was plotted against the number of transactions. This dataset was preprocessed using PCA, and the two most descriminative components were visualized.

## Data Selection
Since the data is `highly Unbalanced` We need to undersample the data. We chose to undersample the majority class, which corresponds to the non-fraudulent class, because we are working on cpu-based systems with limited computational capabilities. Undersampling allows for faster testing and the tradeoffs were found to be negligible.

## Model Hyperparameter Selection
For both logistic regression and SVM, a form of hyperparameter selection was used to find an optimal model. For Logistic regression we used 10-fold cross-validation, and for SVM we used grid search for each of the following kernels: Linear, Radial Basis Function, and Polynomial.

## Performance Analysis
Because of the class imbalance, Recall and Precision were used as the primary performance metrics. For visualization, Confusion Matrices and ROC plots were computed and displayed.
