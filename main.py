# Load the libraries
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Load and Prepare the Data
data = pd.read_csv('Data_for_ML.csv')
X = data[['A', 'B', 'C', 'D']]
y = data['Target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Implement Cross-Validation
n_kf_splits = 5
n_ss_splits = 20
shuffle_split = ShuffleSplit(n_splits=n_ss_splits, test_size=0.2, random_state=42)

# Streamlit web app
st.title("Model Predictions for New Entries")

# User input for new data
st.sidebar.title("New Input Data")
new_input = []
for feature in ['A', 'B', 'C', 'D']:
    new_input.append(st.sidebar.number_input(f"Enter value for {feature}", value=0.0))

# Check if all new input values are zeros
all_zeros = all(val == 0.0 for val in new_input)

if all_zeros:
    st.write("Please enter non-zero values for A, B, C, and D to receive meaningful predictions.")
else:
    # Model Parameters
    model = SVR()
    param_grid = {
        'kernel': ['linear', 'poly', 'rbf'],
        'C': [0.1, 0.2, 0.5, 0.8, 1, 2, 3, 5, 8, 10],
        'epsilon': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0],
    }

    # Train Models and Predict for the new input
    bestmodel_predictions = []

    for train_index, _ in shuffle_split.split(X_scaled):
        X_train, y_train = X_scaled[train_index], y.iloc[train_index]

        # Hyperparameter Optimization
        grid_search = GridSearchCV(model, param_grid, cv=KFold(n_splits=n_kf_splits, shuffle=True), refit=True)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        # Make prediction for the new input using the current best model
        new_input_scaled = scaler.transform([new_input])
        new_prediction = best_model.predict(new_input_scaled)
        bestmodel_predictions.append(new_prediction)

    # Display predictions for the new input
    bestmodel_predictions = np.array(bestmodel_predictions)
    avg_predictions = np.median(bestmodel_predictions, axis=0)
    std_dev = np.std(bestmodel_predictions, axis=0)

    st.write("Predictions for the new input:")
    st.write("Average:", avg_predictions)
    st.write("Standard Deviation:", std_dev)
