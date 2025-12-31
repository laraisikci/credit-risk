import pandas as pd
df = pd.read_csv("credit_risk_dataset.csv")
print(df.dtypes)
print(df.head())
print(df.isnull().sum()) 

