import pandas as pd
import numpy as np
import tensorflow as tf
from keras.layers import Input, GRU, LSTM, Dense, Bidirectional, TimeDistributed, Conv1D, Flatten, Dropout
from keras.models import Model
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy import stats
import time
from tensorflow.keras.layers import Cropping1D

# Load Ethereum block data
data = pd.read_csv('..Data/finalData.csv')
data["baseFees"] /= 1e9
data["gasPrice"] /= 1e9
data['value'] = data['value'].astype(float)
data["value"] /= 1e9
data["maxFeePerGas"] /= 1e9
data["maxPriorityFeePerGas"] /= 1e9


#Remove outliers
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

epochs = data['epoch'].unique()

# Select relevant features and target
features = ['gasUsed','gasPrice','maxFeePerGas','maxPriorityFeePerGas','value','voteCount','activeValidators']  # Adjust to your actual feature columns
target = 'baseFees'

num_epochs = 10  # Number of epochs for each batch training
performance_metrics = []  # Store performance for each epoch and model

def build_lstm_model(input_shape):
    inputs = Input(shape=(input_shape[1], input_shape[2]))
    x = Bidirectional(LSTM(64, return_sequences=True))(inputs)  
    x = Dropout(0.5)(x)
    output = TimeDistributed(Dense(1))(x)  
    model = Model(inputs, output)
    model.compile(optimizer='adam', loss='mse')
    return model

def build_simple_lstm_model(input_shape):
    inputs = Input(shape=(input_shape[1], input_shape[2]))
    x = LSTM(64, return_sequences=True)(inputs)  
    x = Dropout(0.5)(x)
    output = TimeDistributed(Dense(1))(x)  
    model = Model(inputs, output)
    model.compile(optimizer='adam', loss='mse')
    return model

def build_conv_lstm_model(input_shape):
    inputs = Input(shape=(input_shape[1], input_shape[2]))
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = Dropout(0.5)(x)
    x = LSTM(64, return_sequences=True)(x)  
    x = Dropout(0.5)(x)
    output = TimeDistributed(Dense(1))(x)  
    model = Model(inputs, output)
    model.compile(optimizer='adam', loss='mse')
    return model

def build_dense_model(input_shape):
    inputs = Input(shape=(input_shape[1], input_shape[2]))
    x = Dense(128, activation='relu')(inputs)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(1)(x)
    model = Model(inputs, output)
    model.compile(optimizer='adam', loss='mse')
    return model

def build_gru_model(input_shape):
    inputs = Input(shape=(input_shape[1], input_shape[2]))
    x = Bidirectional(GRU(64, return_sequences=True))(inputs)  
    x = Dropout(0.5)(x)
    output = TimeDistributed(Dense(1))(x)  
    model = Model(inputs, output)
    model.compile(optimizer='adam', loss='mse')
    return model

def build_simple_gru_model(input_shape):
    inputs = Input(shape=(input_shape[1], input_shape[2]))
    x = GRU(64, return_sequences=True)(inputs)  
    x = Dropout(0.5)(x)
    output = TimeDistributed(Dense(1))(x)  
    model = Model(inputs, output)
    model.compile(optimizer='adam', loss='mse')
    return model


def build_conv_gru_model(input_shape):
    inputs = Input(shape=(input_shape[1], input_shape[2]))
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = Dropout(0.5)(x)
    x = GRU(64, return_sequences=True)(x)  
    x = Dropout(0.5)(x)
    output = TimeDistributed(Dense(1))(x)  
    model = Model(inputs, output)
    model.compile(optimizer='adam', loss='mse')
    return model


def build_cnn_model(input_shape):
    inputs = Input(shape=(input_shape[1], input_shape[2]))
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    output = Dense(input_shape[1])(x)  
    model = Model(inputs, output)
    model.compile(optimizer='adam', loss='mse')
    return model

# Monte Carlo Dropout uncertainty calculation
def calculate_uncertainty(model, X_val):
    predictions = np.array([model.predict(X_val) for _ in range(10)])
    predictions = predictions.squeeze()
    mean_prediction = predictions.mean(axis=0)
    epistemic_uncertainty = predictions.var(axis=0)
    return mean_prediction, epistemic_uncertainty

def prepare_data(data, features, target, epoch, timesteps=1):
    batch_data = data[data['epoch'] == epoch]
    num_samples = len(batch_data) // timesteps
    X = batch_data[features].values[:num_samples * timesteps].reshape(num_samples, timesteps, len(features))
    y = batch_data[target].values[:num_samples * timesteps].reshape(num_samples, timesteps)

    return X, y


# Function to train models with Monte Carlo Dropout
def train_model_with_dropout(model, X_train, y_train, freeze_layers=True):
    if freeze_layers:
        num_layers = len(model.layers)
        layers_to_freeze = int(0.7 * num_layers)  # Calculate 70% of total layers

        for layer in model.layers[:layers_to_freeze]:  # Freeze the first 70% layers
            layer.trainable = False
        for layer in model.layers[layers_to_freeze:]:  # Keep the rest trainable
            layer.trainable = True
    else:
        for layer in model.layers:
            layer.trainable = True

    model.fit(X_train, y_train, epochs=num_epochs, verbose=1)
    return model


# Performance evaluation function
def evaluate_and_store_performance(model, model_name, X_val, y_val, epoch, performance_metrics):
    start_time = time.time()

    # Estimate epistemic uncertainty using Monte Carlo Dropout
    y_pred, epistemic_uncertainty = calculate_uncertainty(model, X_val)
    prediction_time = (time.time() - start_time) / len(X_val) 
    model_mse = mse(y_val, y_pred)

    performance_metrics.append({
        'epoch': epoch,
        'model': model_name,
        'mse': model_mse,
        'prediction_time': prediction_time,
        'epistemic_uncertainty': epistemic_uncertainty.mean() 
    })

# Main simulation loop for real-time block arrivals
def train_real_time_simulation(data, features, target, freeze_layers):
    shape = (1, 1, len(features))
    models = [
    build_gru_model(shape),
    build_simple_gru_model(shape),
    build_lstm_model(shape),
    build_simple_lstm_model(shape),
    build_dense_model(shape),
    build_conv_lstm_model(shape),
    build_conv_gru_model(shape),
    build_cnn_model(shape)
]

    model_names = [
        'GRU_Model',
        'Simple_GRU_Model',
        'LSTM_Model',
        'Simple_LSTM_Model',
        'Dense_Model',
        'Conv_LSTM_Model',
        'Conv_GRU_Model',
        'CNN_Model'
    ]

    for epoch in epochs:
        print(f"Training on Epoch {epoch} data...")
        X_train, y_train = prepare_data(data, features, target, epoch)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3)

        for i, model in enumerate(models):
            print(f"Training {model_names[i]}")
            models[i] = train_model_with_dropout(model, X_train, y_train, freeze_layers=freeze_layers)
            evaluate_and_store_performance(models[i], model_names[i], X_val, y_val, epoch=epoch, performance_metrics=performance_metrics)

        epoch += 1

train_real_time_simulation(data, features, target, freeze_layers=True)

results = pd.DataFrame(performance_metrics)
results.to_csv('results(baseFee).csv', index=False)
