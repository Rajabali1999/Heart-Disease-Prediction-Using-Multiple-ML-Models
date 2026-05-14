# 🫀 Heart Disease Prediction using Machine Learning

## 📌 Project Overview
This project focuses on predicting the presence of heart disease using machine learning techniques based on 14 clinical features. The workflow includes exploratory data analysis (EDA), preprocessing, handling class imbalance, model training, evaluation, and interpretation.

The goal is to build an accurate and interpretable classification model that can support medical decision-making.

---

## 🎯 Objectives
- Perform exploratory data analysis to understand feature distributions and relationships
- Preprocess data and handle multicollinearity and scaling issues
- Address class imbalance using SMOTE and stratified sampling
- Build and compare multiple machine learning models
- Evaluate models using multiple performance metrics
- Interpret results for medical relevance

---

## 📊 Dataset Description
The dataset contains 14 clinical attributes such as:
- Age
- Sex
- Chest Pain Type (cp)
- Resting Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Maximum Heart Rate (thalach)
- Exercise Induced Angina
- ST Depression (oldpeak)
- And others

**Target Variable:**
- 0 → No Heart Disease  
- 1 → Heart Disease Present  

---

## 🧠 Machine Learning Models Used
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest Classifier (Tree-based Ensemble)
- Dummy Classifier (Baseline)

---

## ⚙️ Methodology
1. Data Loading  
2. Exploratory Data Analysis (EDA)  
3. Data Cleaning and Preprocessing  
4. Feature Scaling (Standardization)  
5. Handling Class Imbalance (SMOTE)  
6. Train-Test Split (Stratified)  
7. Model Training  
8. Hyperparameter Tuning (GridSearchCV)  
9. Model Evaluation  
10. Interpretation of Results  

---

## 📈 Evaluation Metrics
Models were evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

---

## 🏆 Results Summary

| Model            | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|------------------|----------|-----------|--------|----------|---------|
| Random Forest    | 1.000    | 1.000     | 1.000  | 1.000    | 1.000   |
| SVM              | 0.927    | 0.925     | 0.933  | 0.929    | 0.977   |
| Logistic Regression | 0.810 | 0.762     | 0.914  | 0.831    | 0.929   |
| Dummy Classifier | 0.488    | 0.000     | 0.000  | 0.000    | 0.500   |

---

## 📊 Visualizations Included
- Class distribution plots
- Correlation heatmap
- Age distribution analysis
- Chest pain type analysis
- ROC and Precision-Recall curves
- Confusion matrix
- Feature importance (Random Forest)
- Full EDA dashboard

---

## 📌 Key Findings
- Chest pain type, ST depression, and maximum heart rate are strong predictors of heart disease
- Ensemble models (Random Forest) performed best overall
- Logistic Regression provides good interpretability for medical use
- SMOTE improved minority class prediction performance

---

## ⚠️ Limitations
- Dataset size is relatively small
- Data comes from a single source
- Some models assume linear relationships
- Synthetic oversampling may introduce noise

---

## 🚀 Future Improvements
- Apply k-fold cross-validation
- Use advanced ensemble methods (XGBoost, LightGBM)
- Feature engineering for deeper clinical insights
- Validate model on external datasets
- Deploy model using Flask or Streamlit

---

## 🛠️ Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Matplotlib, Seaborn

---

## 👨‍💻 Author
Rajab Ali
