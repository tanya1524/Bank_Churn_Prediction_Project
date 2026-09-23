import pandas as pd  
import pickle 
import matplotlib.pyplot as plt 

x_test =pd.read_csv("D:/Bank_Churn_Project/x_test.csv")

with open("D:/Bank_Churn_Project/random_forest.pkl","rb") as f:
    model=pickle.load(f)

importances = model.feature_importances_ 
feature_names = x_test.columns  

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance(%)" : ((importances * 100).round(2))
}).sort_values(by="Importance(%)",ascending=False) 

print(importance_df)

top10 = importance_df.head(10)

plt.barh(top10["Feature"], top10["Importance(%)"])
plt.xlabel("Importance(%)")
plt.title("Top 10 Features Driving Churn")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("feature_importance.png")

print("\nSaved: feature_importance.png")