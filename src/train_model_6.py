import pandas as pd 
import pickle 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

x_train = pd.read_csv("D:/Bank_Churn_Project/x_train.csv")
x_test = pd.read_csv("D:/Bank_Churn_Project/x_test.csv")
y_train = pd.read_csv("D:/Bank_Churn_Project/y_train.csv").squeeze()
y_test = pd.read_csv("D:/Bank_Churn_Project/y_test.csv").squeeze()

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree" : DecisionTreeClassifier(max_depth=6),
    "Random Forest" : RandomForestClassifier(n_estimators=200),
}

for name , model in models.items():
    model.fit(x_train,y_train)
    print(f"{name} trained successfully.")

    filename = name.lower().replace("","") + ".pkl"
    with open(filename,"wb") as f:
        pickle.dump(model,f)
    print(f"Saved:{filename}")  

      

