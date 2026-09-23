import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

df=pd.read_csv("D:/Bank_Churn_Project/data_features.csv")

x=df.drop(columns=["Exited"])
y=df["Exited"]

x_train, x_test ,y_train , y_test =train_test_split(x,y,test_size=0.2,random_state=42)

scaler =StandardScaler()
x_train=pd.DataFrame(scaler.fit_transform(x_train),columns=x_train.columns)
x_test=pd.DataFrame(scaler.transform(x_test),columns=x_test.columns)

x_train.to_csv("x_train.csv",index=False)
x_test.to_csv("x_test.csv",index=False)
y_train.to_csv("y_train.csv",index=False)
y_test.to_csv("y_test.csv",index=False)


with open("scaler.pkl","wb") as f:
    pickle.dump(scaler,f)
print("Saved: scaler.pkl")


print("Done!")
print("Training rows:",len(x_train),"| Testing rows:",len(x_test))

