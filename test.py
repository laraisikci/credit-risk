import pandas as pd
df = pd.read_csv("credit_risk_dataset.csv")
print(df.dtypes)
import matplotlib.pyplot as plt
plt.scatter(df['person_age'], df['loan_amnt'], c="blue", alpha=0.5)
plt.show()
