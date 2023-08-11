import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

# Load the saved best models
n_ss_splits = 20
best_models = []
for i in range(n_ss_splits):
    model_filename = f"best_model_svr_{i}.joblib"  # Adjust the filename format
    best_models.append(joblib.load(model_filename))

# Create a Streamlit web app
st.title("SVR Model Prediction with Best Models")

# Input form for user to provide 4 input features
st.write("Enter the input features:")
input_features = []
for i in range(4):
    feature = st.number_input(f"Feature {i+1}", value=0.0, format="%.2f")
    input_features.append(feature)

# Normalize the input features
scaler = StandardScaler()
input_features_scaled = scaler.fit_transform([input_features])

# Make predictions using each best model
predictions = np.array([model.predict(input_features_scaled) for model in best_models])

# Display predictions for each best model
st.write("Predictions from each best model:")
for i, pred in enumerate(predictions):
    st.write(f"Best Model {i+1}: {pred[0]:.2f}")

# Calculate and display the average and standard deviation of predictions
average_pred = np.mean(predictions)
std_deviation_pred = np.std(predictions)
st.write(f"Average Prediction: {average_pred:.2f}")
st.write(f"Standard Deviation: {std_deviation_pred:.2f}")
