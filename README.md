# Credit Risk Prediction
A machine learning project that predicts the likelihood of a borrower defaulting on a loan — a core problem in consumer lending, credit scoring, and financial risk management.

## Overview
Credit risk assessment is one of the most important tasks in finance. Lenders need to decide who to give loans to, at what interest rate, and how much — all based on the probability that a borrower will default. This project builds and compares multiple classification models to predict loan default using real-world borrower data.

## Dataset
- **Source:** [Credit Risk Dataset – Kaggle](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
- **Size:** ~32,000 loan records
- **Target variable:** `loan_status` (0 = no default, 1 = default)
- **Features:** borrower age, income, employment length, home ownership, loan amount, interest rate, loan intent, loan grade, credit history length, and prior defaults

## Project Structure
```
credit-risk/
│
├── code.py                  # Full analysis and modeling script
├── credit_risk_dataset.csv  # Dataset
│
└── plots/                   # Generated visualizations
    ├── loan_amount_distribution.png
    ├── income_vs_loan_amount.png
    ├── income_by_loan_status.png
    ├── roc_curves.png
    ├── confusion_matrices.png
    └── feature_importance.png
```
## Methodology
### 1. Exploratory Data Analysis
Analyzed distributions, relationships between key variables, and default rates across borrower segments such as loan intent and home ownership type.

### 2. Data Cleaning
- Removed unrealistic values (age < 18 or > 100, employment length < 0 or > 60)
- Imputed missing numerical values with the median
- Filled missing categorical values with `OTHER`

### 3. Feature Engineering
Encoded categorical variables (`loan_grade`, `loan_intent`, `home_ownership`, `cb_person_default_on_file`) using one-hot encoding and scaled numerical features with `StandardScaler`.

### 4. Class Imbalance Handling
The dataset is imbalanced (~78% non-default, ~22% default). To prevent the models from ignoring the minority class:
- Logistic Regression & Random Forest: `class_weight='balanced'`
- XGBoost: `scale_pos_weight` set to the negative/positive class ratio

### 5. Models Trained
- Logistic Regression (baseline)
- Random Forest
- XGBoost

## Results

| Model | AUC | Accuracy | Precision (default) | Recall (default) | F1 (default) |
|---|---|---|---|---|---|
| Logistic Regression | 0.887 | 82% | 0.57 | 0.79 | 0.66 |
| Random Forest | 0.938 | 93% | 0.96 | 0.72 | 0.82 |
| **XGBoost** | **0.952** | **92%** | **0.82** | **0.79** | **0.81** |

**XGBoost is the best performing model** with an AUC of 0.952 and the strongest balance between precision and recall on the default class. In credit risk, recall is especially important — missing an actual default is more costly than a false alarm — and XGBoost achieves 79% recall while keeping precision high at 82%.

## Key Visualizations

### ROC Curve Comparison
![ROC Curves](plots/roc_curves.png)

### Confusion Matrices
![Confusion Matrices](plots/confusion_matrices.png)

### Feature Importance (Random Forest)
![Feature Importance](plots/feature_importance.png)

### Loan Amount Distribution
![Loan Amount Distribution](plots/loan_amount_distribution.png)

## Key Findings

- **Loan grade and interest rate** are the strongest predictors of default — borrowers with lower grades and higher rates default at significantly higher rates
- **Renters default more than homeowners** — RENT borrowers account for the majority of defaults in the dataset
- **Loan percent income** (how large the loan is relative to income) is a critical risk signal
- **Prior defaults** (`cb_person_default_on_file`) are a strong predictor of future defaults
