import pandas as pd 

import matplotlib.pyplot as plt

#Loading the data

df=pd.read_csv("src/data.csv")

# Overall churn rate - what % of customer left

print("Overall Churn rate:")
print((df["Exited"].value_counts(normalize=True)*100).round(2))

# Churn rate by categories

print("\nChurn rate by Geography:")
print(df.groupby("Geography")["Exited"].mean()*100)

print("\nChurn rate by Gender:")
print(df.groupby("Gender")["Exited"].mean()*100)

print("\nChurn rate by Number of products:")
print(df.groupby("NumOfProducts")["Exited"].mean()*100)

print("\nChurn rate by Active Member Status:")
print(df.groupby("IsActiveMember")["Exited"].mean()*100)

#Saving some charts as image files , so we can look at them

df[["Age","Balance","CreditScore","EstimatedSalary"]].hist(figsize=(10,8))  # draw histogram 
plt.tight_layout()
plt.savefig("distributions.png")
print("\nSaved chart:distributions.png")




