import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
df = pd.read_csv("credit_risk_dataset.csv")

# Display the data types of each column in the DataFrame
print(df.dtypes)

# Display the first few rows of the DataFrame
print(df.head())

# Looking at the distribution of loan amounts with histogram
n, bins, patches = plt.hist(df['loan_amnt'], bins="auto", color='red', 
                            alpha=0.7, rwidth=0.85)
plt.xlabel('Loan Amount')
plt.ylabel('Frequency')
plt.title('Distribution of Loan Amounts')
plt.show()

# Looking at the relationship between income and loan amount with scatter plot
plt.scatter(df['person_income'], df['loan_amnt'], c="blue", alpha=0.5)
plt.xlabel('Annual Income')
plt.ylabel('Loan Amount')
plt.title('Income vs Loan Amount')
plt.show()

# Creating a cross table of the loan intent and loan status
loan_intent_status = pd.crosstab(df['loan_intent'], df['loan_status'], margins=True)
print(loan_intent_status)

# Creating a cross table of the home ownership and loan status
home_ownership_status = pd.crosstab(df['person_home_ownership'], df['loan_status'], margins=True)
print(home_ownership_status)

# Creating a box plot to visualize the distribution of income based on loan status
df.boxplot(column='person_income', by='loan_status', grid=False, color='green')
plt.title('Income Distribution by Loan Status')
plt.suptitle('')
plt.xlabel('Loan Status')
plt.ylabel('Annual Income')
plt.show()

# Starting with data cleaning
# Since an employment length of less than 0 and more than 60 is not realistic, 
# we will remove those rows
indices = df[(df['person_emp_length'] < 0) | (df['person_emp_length'] > 60)].index
df_new = df.drop(indices)
# Since age less than 18 and more than 100 are not realistic, we will remove 
# those rows
indices_age = df_new[(df_new['person_age'] < 18) | (df_new['person_age'] > 100)].index
df_cleaned_age = df_new.drop(indices_age)

# Replacing missing data 
print(df_cleaned_age.isnull().sum()) 
df_cleaned_age['person_emp_length'] = df_cleaned_age['person_emp_length'].fillna(df_cleaned_age['person_emp_length'].median())
df_cleaned_age['loan_int_rate'] = df_cleaned_age['loan_int_rate'].fillna(df_cleaned_age['loan_int_rate'].median())
print(df_cleaned_age.isnull().sum()) 
print(df_cleaned_age['person_home_ownership'].value_counts())

# Replacing missing categorical data with other
df_cleaned_age['person_home_ownership'] = df_cleaned_age['person_home_ownership'].fillna('OTHER')
print(df_cleaned_age.isnull().sum())

# Encoding categorical variables 
df_model = pd.get_dummies(df_cleaned_age, columns=[
    'person_home_ownership',
    'loan_intent',
    'loan_grade',
    'cb_person_default_on_file'
], drop_first=True)

# Feature selection
feature_cols = [
    'loan_int_rate', 'person_emp_length', 'person_income',
    'loan_amnt', 'loan_percent_income', 'person_age', 'cb_person_cred_hist_length'
] + [col for col in df_model.columns if col.startswith((
    'person_home_ownership_',
    'loan_intent_',
    'loan_grade_',
    'cb_person_default_on_file_'
))]


# Creating X and y for modeling
X = df_model[feature_cols]
y = df_model['loan_status']


# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y,   test_size=0.2, 
                                                    random_state=123)

#Feature scaling 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   
X_test_scaled = scaler.transform(X_test)  

#Logistic Regression
clf_logistic = LogisticRegression(solver="lbfgs", max_iter=1000, class_weight="balanced")
clf_logistic.fit(X_train_scaled, y_train)

preds = clf_logistic.predict(X_test_scaled)
print(" Logistic Regression ")
print(classification_report(y_test, preds))

#Random Forest 
from sklearn.ensemble import RandomForestClassifier

clf_rf = RandomForestClassifier(n_estimators=100, random_state=123, n_jobs=-1, class_weight="balanced")
clf_rf.fit(X_train, y_train)

#XGBoost 
from xgboost import XGBClassifier
neg, pos = y_train.value_counts()[0], y_train.value_counts()[1]
scale = neg / pos

clf_xgb = XGBClassifier(n_estimators=100, random_state=123,
                         use_label_encoder=False, eval_metric='logloss', scale_pos_weight = scale)
clf_xgb.fit(X_train, y_train)

#Evaluating the models with the models 
models = {
    "Logistic Regression": (clf_logistic, X_test_scaled),
    "Random Forest":       (clf_rf,       X_test),
    "XGBoost":             (clf_xgb,      X_test),
}

for name, (model, X_eval) in models.items():
    preds = model.predict(X_eval)
    print(f" {name} ")
    print(classification_report(y_test, preds))

#ROC/AUC Curves
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, ConfusionMatrixDisplay


plt.figure(figsize=(8, 6))

for name, (model, X_eval) in models.items():
    proba = model.predict_proba(X_eval)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

plt.plot([0, 1], [0, 1], 'k--', label="Random baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.show() 

#Confusion Matrix 
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, ConfusionMatrixDisplay

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, (name, (model, X_eval)) in zip(axes, models.items()):
    preds = model.predict(X_eval)
    ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=ax, colorbar=False)
    ax.set_title(name)

plt.tight_layout()
plt.show()

#Feature Importance
importances = pd.Series(clf_rf.feature_importances_, index=feature_cols)
top15 = importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(8, 6))
top15.sort_values().plot(kind='barh', color='steelblue')
plt.title("Top 15 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

