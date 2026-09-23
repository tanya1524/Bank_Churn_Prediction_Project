import streamlit as st 
import pandas as pd  
import pickle  

with open("random_forest.pkl","rb") as f:
    model = pickle.load(f)

with open("scaler.pkl","rb") as f:
   scaler = pickle.load(f)   

x_train_columns = pd.read_csv("x_train.csv").columns.tolist()   

st.title("Bank Customer Churn Predictor")  
st.write("Enter a customer's details to see their churn. ")


credit_score = st.slider("Credit Score", 350,850,650)
age = st.slider ("Age",18,92,40)
tenure= st.slider("Tenure(years with bank)", 0,10,5)
balance= st.number_input("Balance",0.0,250000.0,50000.0)
num_products =st.selectbox("Number of Products",[1,2,3,4])
salary= st.number_input("Estimated Salary", 0.0,200000.0,60000.0)
is_active = st.radio("Active Member?",["Yes","No"])
has_cr_card= st.radio("Has Credit Card?",["Yes","No"])
geography = st.selectbox("Geography",["France","Germany","Spain"])
gender=st.selectbox("Gender",["Male","Female"])


if st.button("Predict Churn Risk"):  

     balance_salary_ratio = balance / salary if salary !=0 else 0
     product_density = num_products / (tenure + 1)
     engagement_product = (1 if is_active == "Yes" else 0) * num_products 
     age_tenure = age * tenure  
     has_zero_balance = 1 if balance == 0 else 0



     input_data =pd.DataFrame([{
    
         "CreditScore": credit_score,
         "Age": age,
         "Tenure": tenure,
         "Balance": balance,
         "NumOfProducts": num_products,
         "HasCrCard": 1 if has_cr_card == "Yes" else 0 ,
         "IsActiveMember": 1 if is_active =="Yes" else 0,
         "EstimatedSalary": salary,
         "BalanceSalaryRatio": balance_salary_ratio,
         "ProductDensity": product_density,
         "EngagementProductInteraction": engagement_product,
         "AgeTenureInteraction": age_tenure,
         "HasZeroBalance": has_zero_balance,
         "Geography_France": 1 if geography == "France" else 0 ,
         "Geography_Germany": 1 if geography == "Germany" else 0,
         "Geography_Spain": 1 if geography == "Spain" else 0,
         "Gender_Female": 1 if gender =="Female" else 0,
         "Gender_Male": 1 if gender =="Male" else 0,

}]) 

    
     input_data = input_data[x_train_columns]

     input_data = pd.DataFrame(scaler.transform(input_data), columns=x_train_columns)

     proba = model.predict_proba(input_data)[0][1]

     st.subheader(f"Churn Probability: {proba * 100:.1f}%")

     if proba >= 0.6:
        st.error("High Risk of Churn")
     elif proba >=0.3:
        st.error("Medium Risk of Churn")
     else : 
        st.success("Low Risk of Churn") 


           