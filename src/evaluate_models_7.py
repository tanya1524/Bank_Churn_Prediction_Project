import pandas as pd 

import pickle 
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score

x_test = pd.read_csv("D:/Bank_Churn_Project/x_test.csv")
y_test = pd.read_csv("D:/Bank_Churn_Project/y_test.csv").squeeze()

model_files={
    "Logistic Regression" : "D:/Bank_Churn_Project/logistic_regression.pkl",
    "Decision Tree" : "D:/Bank_Churn_Project/decision_tree.pkl",
    "Random Forest" : "D:/Bank_Churn_Project/random_forest.pkl",
}

for name , filename in model_files.items():
    with open(filename,"rb") as f:
        model = pickle.load(f)  

    proba =model.predict_proba(x_test)[:,1] 

    pred=(proba>=0.5).astype(int)

    print(f"\n---{name}---")
    print("Accuracy:",round(accuracy_score(y_test,pred),3)*100)
    print("Precision:",round(precision_score(y_test,pred),3)*100)
    print("Recall:",round(recall_score(y_test,pred),3)*100)
    print("F1 Score:",round(f1_score(y_test,pred),3)*100)
    print("ROC-AUC:",round(roc_auc_score(y_test,pred),3)*100)