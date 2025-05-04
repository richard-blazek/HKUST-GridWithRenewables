# Dataset: https://www.kaggle.com/datasets/mubashirrahim/wind-power-generation-data-forecasting

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

def load_setup():
    df = pd.read_csv('Power.csv')
    df['hour'] = df['Time'].apply(lambda t: int(t[-8:-6]))
    return df

def split_data(df: pd.DataFrame, pred_len: int):
    return df[:-pred_len], df[-pred_len:]

def extract_xy(df: pd.DataFrame):
    X = df[['hour','temperature_2m','relativehumidity_2m','windspeed_10m','windspeed_100m','winddirection_10m','winddirection_100m','windgusts_10m','Power']][:-2]
    y = df['Power'][2:]
    return X, y

def scale_data(X_train, X_pred):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_pred = scaler.transform(X_pred)
    return X_train, X_pred

def train_mlp_regressor(X_train, y_train):
    model = MLPRegressor((128, 64, 32), max_iter=10000, random_state=42, early_stopping=True)
    model.fit(X_train, y_train)
    return model

df = load_setup()
df_train, df_pred = split_data(df, 720)

X_train, y_train = extract_xy(df_train)
X_pred, y_pred = extract_xy(df_pred)
X_train, X_pred = scale_data(X_train, X_pred)

regressor = train_mlp_regressor(X_train, y_train)

y_hat = regressor.predict(X_pred)
print('R-squared:', r2_score(y_pred, y_hat))

plt.plot(list(zip(y_pred, y_hat)))
plt.show()
