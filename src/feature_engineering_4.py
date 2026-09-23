import pandas as pd 

df=pd.read_csv("D:/Bank_Churn_Project/data-cleaned.csv")

df["BalanceSalaryRatio"]=df["Balance"]/ df["EstimatedSalary"].replace(0,1)

df["ProductDensity"]= df["NumOfProducts"]/ (df["Tenure"]+1)

df["EngagementProductInteraction"] = df["IsActiveMember"] * df["NumOfProducts"]

df["AgeTenureInteraction"] = df["Age"] * df["Tenure"]

df["HasZeroBalance"] = (df["Balance"] == 0).astype(int)

print("New columns added:")
print(df[["BalanceSalaryRatio","ProductDensity","EngagementProductInteraction","AgeTenureInteraction","HasZeroBalance"]].head())

df.to_csv("data_features.csv", index=False)
print("\nSaved: data_features.csv")

