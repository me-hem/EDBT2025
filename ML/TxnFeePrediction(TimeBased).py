import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from scipy import stats


# Load Ethereum block data 
data = pd.read_csv('../Data/finalData.csv')
data["baseFees"] /= 1e9
data["gasPrice"] /= 1e9
data['value'] = data['value'].astype(float)
data["value"] /= 1e9
data["maxFeePerGas"] /= 1e9
data["maxPriorityFeePerGas"] /= 1e9
data["transactionFee"] /= 1e9


data['timestamp'] = pd.to_datetime(data['timestamp'])
data.set_index('timestamp', inplace=True)
# Extract time-based features
data['hour'] = data.index.hour
data['day_of_week'] = data.index.dayofweek
# Drop the original timestamp column
data.reset_index(drop=True, inplace=True)

# Remove outliers
data = data[~(data == -1).any(axis=1)]
z_scores = np.abs(stats.zscore(data))
outliers = (z_scores > 3).any(axis=1)
data = data[~outliers]
multiplier = 1.5
data_no_outliers = data.copy()

for column in data.columns:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    data_no_outliers = data_no_outliers[~data_no_outliers.index.isin(outliers.index)]
data = data_no_outliers



# Select the relevant columns
features = ['gasUsed', 'maxFeePerGas', 'maxPriorityFeePerGas', 'value', 'activeValidators', 'hour', 'day_of_week']
target = 'transactionFee'

# Split the dataset into features and target
X = data[features]
y = data[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the data 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
mlp_model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500, random_state=42)
svm_model = SVR(kernel='rbf', C=100, epsilon=0.1)
lgbm_model = LGBMRegressor(n_estimators=100, learning_rate=0.1, random_state=42)

# Train the models
print("Training model: Random Forest")
rf_model.fit(X_train, y_train)


print("Training model: XGBoost")
xgb_model.fit(X_train, y_train)

print("Training model: MLP")
mlp_model.fit(X_train_scaled, y_train)  # For neural network we use scaled data

print("Training model: SVM")
svm_model.fit(X_train_scaled, y_train)  # For SVM we use scaled data

print("Training model: LightGBM")
lgbm_model.fit(X_train, y_train)

# Predictions
rf_pred = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)
mlp_pred = mlp_model.predict(X_test_scaled)
svm_pred = svm_model.predict(X_test_scaled)
lgbm_pred = lgbm_model.predict(X_test)

# Evaluation: MSE and MAE
def evaluate_model(y_test, predictions):
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    return mse, mae

# Store predictions and metrics in CSV files
results = pd.DataFrame({
    'Actual': y_test,
    'RF_Prediction': rf_pred,
    'XGB_Prediction': xgb_pred,
    'MLP_Prediction': mlp_pred,
    'SVM_Prediction': svm_pred,
    'LGBM_Prediction': lgbm_pred
})

# Save the actual vs prediction results to a CSV file
results.to_csv('txnFeeResults.csv', index=False)
print("Actual vs Predicted results saved to 'txnFeeResults.csv'")


# Store error metrics in a dictionary
metrics = {
    'Model': ['Random Forest', 'XGBoost', 'MLP', 'SVM', 'LightGBM'],
    'MSE': [
        mean_squared_error(y_test, rf_pred),
        mean_squared_error(y_test, xgb_pred),
        mean_squared_error(y_test, mlp_pred),
        mean_squared_error(y_test, svm_pred),
        mean_squared_error(y_test, lgbm_pred)
    ],
    'MAE': [
        mean_absolute_error(y_test, rf_pred),
        mean_absolute_error(y_test, xgb_pred),
        mean_absolute_error(y_test, mlp_pred),
        mean_absolute_error(y_test, svm_pred),
        mean_absolute_error(y_test, lgbm_pred)
    ]
}

metrics_df = pd.DataFrame(metrics)

# Save the error metrics to a CSV file
metrics_df.to_csv('txnFeeMetrics.csv', index=False)
print("Model metrics saved to 'txnFeeMetrics.csv'")

