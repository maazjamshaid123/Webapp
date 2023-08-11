import streamlit as st
import joblib
import numpy as np

# Load the saved best models
n_ss_splits = 20
best_models = []
for i in range(n_ss_splits):
    filename = f"best_model_svr_{i}.joblib"  # Adjust the filename format as needed
    best_model = joblib.load(filename)
    best_models.append(best_model)

# Function to make predictions using each best model
def make_predictions(input_features):
    predictions = []
    for model in best_models:
        prediction = model.predict(input_features.reshape(1, -1))
        predictions.append(prediction[0])
    return predictions

# Streamlit web app
st.title("SVR Prediction ")

# Input fields for features
input_features = []
for feature in ['A', 'B', 'C', 'D']:
    value = st.sidebar.number_input(f"Enter value for {feature}", step=0.01)
    input_features.append(value)
input_features = np.array(input_features)

# Make predictions using each best model
predictions = make_predictions(input_features)

# Display individual predictions
st.header("Individual Predictions from Each Best Model")
num_columns = 3
with st.container():
    col_width = 12 // num_columns
    for i, prediction in enumerate(predictions):
        if i % num_columns == 0:
            col = st.columns(num_columns)
        col[i % num_columns].info(f"Model {i+1}: {prediction:.2f}")
    

# Calculate and display average and standard deviation
average_prediction = np.mean(predictions)
std_deviation = np.std(predictions)
st.header("Aggregate Results")
st.error(f"Average Prediction: {average_prediction:.2f}")
st.error(f"Standard Deviation: {std_deviation:.2f}")
