import pandas as pd 

df=pd.read_csv("src/data.csv")

for col in ["CreditScore","Balance","EstimatedSalary"]:
    if col in df.columns:
        df[col]=df[col].fillna(df[col].median())

for col in ["Geography","Gender"]:     
    if col in df.columns:
        df[col]=df[col].fillna(df[col].mode()[0])

df.drop(["CustomerId","Surname","Year"],axis=1,inplace=True)     

df=pd.get_dummies(df,columns=["Geography","Gender"])     

print("Missing values left:")
print(df.isnull().sum())

print("\nColumns after cleaning:")
print(df.columns.tolist())

df.to_csv(("data-cleaned.csv"),index=False)

print("\nSaved : data_cleaned.csv")