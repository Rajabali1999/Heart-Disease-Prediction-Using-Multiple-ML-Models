# Heart Disease Prediction Using Multiple ML-Models
Advanced machine learning project for heart disease prediction using the UCI dataset. Includes preprocessing, SMOTE, model comparison (Logistic Regression, Random Forest, SVM), hyperparameter tuning, and performance evaluation.

**The analysis includes:**

Data visualization and statistical analysis

Handling class imbalance using SMOTE

Feature selection using Recursive Feature Elimination (RFE)

Training and comparing multiple machine learning models

Hyperparameter tuning of the best-performing model

Model evaluation using accuracy, ROC-AUC, learning curves, and precision-recall curves

# Dataset

**The dataset used is heart.csv containing the following columns:**

Column	Description

age	Age of the patient

sex	Gender (0 = Female, 1 = Male)

cp	Chest pain type (0–3)

trestbps	Resting blood pressure (mm Hg)

chol	Serum cholesterol (mg/dl)

fbs	Fasting blood sugar > 120 mg/dl (1 = True, 0 = False)

restecg	Resting electrocardiographic results (0–2)

thalach	Maximum heart rate achieved

exang	Exercise-induced angina (1 = Yes, 0 = No)

oldpeak	ST depression induced by exercise relative to rest

slope	Slope of the peak exercise ST segment (0–2)

ca	Number of major vessels colored by fluoroscopy (0–3)

thal	Thalassemia (1 = Normal, 2 = Fixed defect, 3 = Reversible defect)

target	Heart disease diagnosis (0 = No, 1 = Yes)

# Requirements #

Install the required libraries:

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV

from sklearn.preprocessing import StandardScaler, LabelEncoder

from sklearn.feature_selection import SelectKBest, chi2, RFE

from imblearn.over_sampling import SMOTE

import warnings

warnings.filterwarnings('ignore')

# 1. Data Loading and Inspection

Load dataset with pandas

Inspect dataset shape, missing values, and basic statistics

Check class distribution for the target variable

# 2. Exploratory Data Analysis (EDA)

Visualizations using matplotlib and seaborn:

Target distribution

Age distribution by target

Gender distribution

Chest pain type vs target

Correlation heatmap

Max heart rate vs target

Oldpeak distribution

Age vs Max Heart Rate scatter plot

# 3. Preprocessing

Split dataset into training and testing sets (80/20)

Feature scaling with StandardScaler

Handle class imbalance using SMOTE

# 4. Feature Selection

Recursive Feature Elimination (RFE) with Random Forest

Select top 8 most important features

# 5. Model Training and Comparison

Models evaluated:

Logistic Regression

Decision Tree

Random Forest

Gradient Boosting

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

XGBoost

Naive Bayes

AdaBoost

Metrics:

Accuracy

ROC-AUC

Cross-validation scores

Visualize model comparison with bar charts

# 6. Hyperparameter Tuning

Tuned Random Forest using GridSearchCV

Parameters tuned: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features

Evaluate the tuned model on the test set

# 7. Advanced Evaluation

Feature importance plot

Confusion matrix

ROC curves for all models

Precision-Recall curves

Learning curves for the best model

# Results

Best Model: Tuned Random Forest

Test Accuracy: ~0.88

Test ROC-AUC: ~0.93

Top Features: thalach, oldpeak, cp

Feature importance and model comparison results are saved as CSV:

feature_importance.csv

model_comparison_results.csv



Run the analysis script:

python heart_disease_analysis.py

Explore generated visualizations and CSV results.

# Visualizations

The project generates:

Distribution plots, boxplots, and scatter plots for feature analysis

Heatmaps for feature correlations

Bar charts comparing model accuracy and ROC-AUC

ROC and Precision-Recall curves for all models

Learning curves for the tuned Random Forest

# Author

Your Name  Rajab Ali
