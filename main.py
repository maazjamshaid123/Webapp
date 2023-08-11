# Import necessary libraries
import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the ShuffleSplit best models
n_ss_splits = 20
best_models = [joblib.load(f"saved_svr_model_{i}.joblib") for i in range(n_ss_splits)]

# Define a function to make predictions using each best model
def predict_multiple_models(input_data, best_models):
    predictions = [model.predict(input_data) for model in best_models]
    return predictions

# Create the Streamlit web app
st.title("SVR Model Predictions")

# Input form for new entry
st.header("Input for Prediction")
input_features = st.text_input("Input features (comma-separated values)", "1,2,3,4")
input_data = np.array([list(map(float, input_features.split(",")))])  # Convert input to a 2D array

# Predictions using each best model from ShuffleSplit
multiple_models_predictions = predict_multiple_models(input_data, best_models)

# Calculate average and standard deviation of predictions
predictions_array = np.array(multiple_models_predictions)
average_predictions = np.mean(predictions_array, axis=0)
std_dev_predictions = np.std(predictions_array, axis=0)

# Display the results
st.subheader("Predictions using each best model from ShuffleSplit")
st.write("Individual Predictions:", multiple_models_predictions)
st.write("Average Prediction:", average_predictions[0])
st.write("Standard Deviation of Predictions:", std_dev_predictions[0])
