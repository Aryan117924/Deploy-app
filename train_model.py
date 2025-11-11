import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
import joblib

# Load dataset
dt = pd.read_csv("creditcard.csv")

# Handle numeric conversion
dt = dt.apply(pd.to_numeric, errors='coerce')
dt.dropna(inplace=True)

# Feature scaling
numerical_cols = dt.select_dtypes(include=['float64', 'int64']).columns
scaler = MinMaxScaler()
dt[numerical_cols] = scaler.fit_transform(dt[numerical_cols])

# Split data
X = dt.drop('Class', axis=1)
y = dt['Class']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(x_train, y_train)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_res, y_res)

# Save model
joblib.dump(model, "credit_fraud.pkl")
print("✅ Model saved as credit_fraud.pkl")
