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

