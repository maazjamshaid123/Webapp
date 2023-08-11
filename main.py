# Load the libraries
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load and Prepare the Data
data = pd.read_csv('Data_for_ML.csv')
X = data[['A', 'B', 'C', 'D']]
y = data['Target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Implement Cross-Validation
n_kf_splits = 5
n_ss_splits = 20
kfold = KFold(n_splits=n_kf_splits, shuffle=True, random_state=42)
shuffle_split = ShuffleSplit(n_splits=n_ss_splits, test_size=0.2, random_state=42)

# Streamlit web app
st.title("Model Predictions and Evaluation")

# User input for new data
st.sidebar.title("New Input Data")
new_input = []
for feature in ['A', 'B', 'C', 'D']:
    new_input.append(st.sidebar.number_input(f"Enter value for {feature}", value=0.0))

# Model Parameters
model = SVR()
param_grid = {
    'kernel': ['linear', 'poly', 'rbf'],
    'C': [0.1, 0.2, 0.5, 0.8, 1, 2, 3, 5, 8, 10],
    'epsilon': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0],
}

# Train and Evaluate Models
model_results = {'Predictions': [], 'Actuals': []}
bestmodel_pred = [[] for _ in range(n_ss_splits)]

for ss_idx, (train_index, test_index) in enumerate(shuffle_split.split(X_scaled)):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Hyperparameter Optimization
    grid_search = GridSearchCV(model, param_grid, cv=kfold, refit=True)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    model_results['Predictions'].extend(y_pred)
    model_results['Actuals'].extend(y_test)

    # Make predictions for the new input
    new_input_scaled = scaler.transform([new_input])
    new_prediction = best_model.predict(new_input_scaled)
    bestmodel_pred[ss_idx].extend(new_prediction)

# Display average and standard deviation of predictions for the new input
new_predictions = np.array(bestmodel_pred).flatten()
st.write("Predictions for the new input:")
st.write("Average:", np.mean(new_predictions))
st.write("Standard Deviation:", np.std(new_predictions))

# Compute model evaluation metrics
actuals = np.array(model_results['Actuals'])
predictions = np.array(model_results['Predictions'])
r2 = r2_score(actuals, predictions)
mae = mean_absolute_error(actuals, predictions)
mse = mean_squared_error(actuals, predictions)

# Display evaluation metrics
st.header("Model Evaluation:")
st.write("R-squared:", r2)
st.write("Mean Absolute Error (MAE):", mae)
st.write("Mean Squared Error (MSE):", mse)
