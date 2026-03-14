

# Heart Disease Prediction Using Multiple ML Models

This project applies machine learning techniques to predict the presence of heart disease using the **UCI Heart Disease Dataset**. The workflow includes data preprocessing, handling class imbalance, feature selection, model comparison, hyperparameter tuning, and advanced evaluation.

The implementation is developed using **Python** with machine learning tools from **Scikit-learn** and **XGBoost**.

---

# Project Features

The analysis includes:

* Data visualization and statistical analysis
* Handling class imbalance using **SMOTE**
* Feature selection using **Recursive Feature Elimination** (RFE)
* Training and comparing multiple machine learning models
* Hyperparameter tuning of the best-performing model
* Model evaluation using accuracy, ROC-AUC, learning curves, and precision-recall curves

---

# Dataset

The dataset used in this project is **heart.csv**, containing medical attributes used to diagnose heart disease.

| Column   | Description                                    |
| -------- | ---------------------------------------------- |
| age      | Age of the patient                             |
| sex      | Gender (0 = Female, 1 = Male)                  |
| cp       | Chest pain type (0–3)                          |
| trestbps | Resting blood pressure (mm Hg)                 |
| chol     | Serum cholesterol (mg/dl)                      |
| fbs      | Fasting blood sugar > 120 mg/dl                |
| restecg  | Resting electrocardiographic results           |
| thalach  | Maximum heart rate achieved                    |
| exang    | Exercise-induced angina                        |
| oldpeak  | ST depression induced by exercise              |
| slope    | Slope of the peak exercise ST segment          |
| ca       | Number of major vessels colored by fluoroscopy |
| thal     | Thalassemia                                    |
| target   | Heart disease diagnosis (0 = No, 1 = Yes)      |

---

# Requirements

Install the required Python libraries:

* **NumPy**
* **Pandas**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **XGBoost**
* **imbalanced-learn**

Example installation:

```
pip install numpy pandas matplotlib seaborn scikit-learn xgboost imbalanced-learn
```

---

# 1. Data Loading and Inspection

* Load the dataset using **Pandas**
* Inspect dataset shape, missing values, and statistical summary
* Analyze class distribution for the target variable

---

# 2. Exploratory Data Analysis (EDA)

Visualizations using **Matplotlib** and **Seaborn**:

* Target distribution
* Age distribution by target
* Gender distribution
* Chest pain type vs target
* Correlation heatmap
* Maximum heart rate vs target
* Oldpeak distribution
* Age vs Maximum Heart Rate scatter plot

---

# 3. Data Preprocessing

Steps performed:

* Split dataset into training and testing sets (**80/20 split**)
* Feature scaling using **Standardization** (`StandardScaler`)
* Handling class imbalance using **SMOTE**

---

# 4. Feature Selection

Feature selection performed using:

**Recursive Feature Elimination (RFE)** with **Random Forest**

Top **8 most important features** were selected for model training.

---

# 5. Model Training and Comparison

Multiple machine learning algorithms were evaluated:

* **Logistic Regression**
* **Decision Tree**
* **Random Forest**
* **Gradient Boosting**
* **Support Vector Machine**
* **K-Nearest Neighbors**
* **XGBoost**
* **Naive Bayes**
* **AdaBoost**

### Evaluation Metrics

* Accuracy
* ROC-AUC
* Cross-validation scores

Model performance was visualized using **bar charts**.

---

# 6. Hyperparameter Tuning

Hyperparameter tuning was applied to **Random Forest** using:

**Grid Search (`GridSearchCV`)**

Parameters tuned:

* `n_estimators`
* `max_depth`
* `min_samples_split`
* `min_samples_leaf`
* `max_features`

The tuned model was evaluated on the test dataset.

---

# 7. Advanced Evaluation

Additional model evaluation techniques include:

* Feature importance plot
* Confusion matrix
* ROC curves for all models
* Precision–Recall curves
* Learning curves for the best model

---

# Results

**Best Model:** Tuned **Random Forest**

* Test Accuracy: ~0.88
* Test ROC-AUC: ~0.93

### Most Important Features

* `thalach`
* `oldpeak`
* `cp`

Generated output files:

* `feature_importance.csv`
* `model_comparison_results.csv`

---

# Running the Project

Run the analysis script:

```
python heart_disease_analysis.py
```

Then explore the generated visualizations and result CSV files.

---

# Visualizations

The project generates:

* Distribution plots, boxplots, and scatter plots for feature analysis
* Correlation heatmaps
* Model comparison bar charts
* ROC and Precision–Recall curves
* Learning curves for the tuned Random Forest model

---

# Author

**Rajab Ali**

