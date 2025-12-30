import pandas as pd
df = pd.read_csv("credit_risk_dataset.csv")

# Display the data types of each column in the DataFrame
print(df.dtypes)

# Display the first few rows of the DataFrame
print(df.head())

# Looking at the distribution of loan amounts with histogram
import matplotlib.pyplot as plt
n, bins, patches = plt.hist(df['loan_amnt'], bins="auto", color='red', 
                            alpha=0.7, rwidth=0.85)
plt.xlabel('Loan Amount')
plt.ylabel('Frequency')
plt.title('Distribution of Loan Amounts')
plt.show()

# Looking at the relationship between income and loan amount with scatter plot
plt.scatter(df['annual_inc'], df['loan_amnt'], c="blue", alpha=0.5)
plt.xlabel('Annual Income')
plt.ylabel('Loan Amount')
plt.title('Income vs Loan Amount')
plt.show()

# Creating a cross table of the loan intent and loan status
loan_intent_status = pd.crosstab(df['loan_intent'], df['loan_status'], margins=True)
print(loan_intent_status)

# Creating a cross table of the home ownership and loan status
home_ownership_status = pd.crosstab(df['home_ownership'], df['loan_status'], margins=True)
print(home_ownership_status)

# Creating a box plot to visualize the distribution of income based on loan status
df.boxplot(column='annual_inc', by='loan_status', grid=False, color='green')
plt.title('Income Distribution by Loan Status')
plt.suptitle('')
plt.xlabel('Loan Status')
plt.ylabel('Annual Income')
plt.show()

# Starting with data cleaning
# Since an employment length of less than 0 and more than 60 is not realistic, 
# we will remove those rows
indices = df[(df['emp_length'] < 0) | (df['emp_length'] > 60)].index
df_new = df.drop(indices)
# Since age less than 18 and more than 100 are not realistic, we will remove 
# those rows
indices_age = df_new[(df_new['age'] < 18) | (df_new['age'] > 100)].index
df_cleaned_age = df_new.drop(indices_age)

# Replacing missing data 
