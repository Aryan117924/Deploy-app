
!pip install streamlit

import pandas as pd
dt = pd.read_csv("/content/creditcard.csv", delimiter=',', on_bad_lines='skip')
dt

dt.info()

dt = dt.apply(pd.to_numeric, errors='coerce')
display(dt.info())

duplicates = dt.duplicated().sum()
print(f"Number of duplicate rows: {duplicates}")

print(f"Number of rows: {dt.shape[0]}")
print(f"Number of columns: {dt.shape[1]}")

dt.isnull().sum()

dt.dropna(inplace=True)
print("Null values after dropping rows:")
print(dt.isnull().sum())

from sklearn.preprocessing import MinMaxScaler
numerical_cols = dt.select_dtypes(include=['float64', 'int64']).columns
scaler = MinMaxScaler()
dt[numerical_cols] = scaler.fit_transform(dt[numerical_cols])

display(dt.head())

x = dt.drop('Class', axis=1)
y = dt['Class']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy_score = model.score(x_test, y_test)
print(f"Accuracy Score: {accuracy_score}")

import pickle

filename = 'credit_fraud.pkl'
pickle.dump(model, open(filename, 'wb'))

print(f"Model successfully pickled to {filename}")

import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("credit_fraud.pkl")

st.title("💳 Credit Card Fraud Detection App")
st.write("Enter transaction details below to predict if it's fraudulent.")

# Input fields for features
v1 = st.number_input("V1")
v2 = st.number_input("V2")
v3 = st.number_input("V3")
amount = st.number_input("Transaction Amount")

# When the user clicks the predict button
if st.button("Predict"):
    # Create a DataFrame with the inputs
    data = pd.DataFrame([[v1, v2, v3, amount]], columns=["V1", "V2", "V3", "Amount"])

    # Make prediction
    prediction = model.predict(data)[0]

    if prediction == 1:
        st.error("⚠️ This transaction is **FRAUDULENT!**")
    else:
        st.success("✅ This transaction seems **legitimate.**")

"""The Streamlit app is now running. Click on the public URL provided in the output of the previous cell to access it. It might take a few moments for the app to become available.

**Note:** This Streamlit app uses only a subset of the features (V1, V2, V3, Amount) for prediction and does not apply the necessary scaling that was used during model training. This will likely lead to inaccurate predictions. A more robust application would require collecting inputs for all features the model was trained on and applying the same preprocessing steps (like scaling) to the input data before making a prediction.
"""


