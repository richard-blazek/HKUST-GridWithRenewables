# Dataset: https://www.kaggle.com/datasets/mubashirrahim/wind-power-generation-data-forecasting

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

def load_setup():
    df = pd.read_csv('Power.csv')
    df['hour'] = df['Time'].apply(lambda t: int(t[-8:-6]))
    return df

def split_data(df: pd.DataFrame, sz1: float, sz2: float):
    pt1 = int(sz1 * len(df.index))
    pt2 = int((sz1 + sz2) * len(df.index))
    return df[:pt1], df[pt1:pt2], df[pt2:]

def extract_xy(df: pd.DataFrame):
    X = df[['hour','temperature_2m','relativehumidity_2m','windspeed_10m','windspeed_100m','winddirection_10m','winddirection_100m','windgusts_10m','Power']][:-2]
    y = df['Power'][2:]
    return X, y

def scale_data(X_train, X_estim, X_sim):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_estim = scaler.transform(X_estim)
    X_sim = scaler.transform(X_sim)
    return X_train, X_estim, X_sim

def train_mlp_regressor(X_train, y_train):
    model = MLPRegressor((64, 32, 16), max_iter=10000, random_state=42, early_stopping=True)
    model.fit(X_train, y_train)
    return model

def estimate_model_statistics(regressor: MLPRegressor, X_estim, y_estim):
    y_hat = regressor.predict(X_estim)
    r2 = r2_score(y_estim, y_hat)
    mse = mean_squared_error(y_estim, y_hat)
    sigma = np.std(y_estim - y_hat, ddof=(regressor.n_features_in_ + 1) * 64 + 65 * 32 + 33 * 16 + 17)
    return r2, mse, sigma

df = load_setup()
df_train, df_estim, df_sim = split_data(df, 0.6, 0.3)

X_train, y_train = extract_xy(df_train)
X_estim, y_estim = extract_xy(df_estim)
X_sim, y_sim = extract_xy(df_sim)
X_train, X_estim, X_sim = scale_data(X_train, X_estim, X_sim)
y_train, y_estim, y_sim = y_train.to_numpy(), y_estim.to_numpy(), y_sim.to_numpy()

regressor = train_mlp_regressor(X_train, y_train)

r2, mse, sigma = estimate_model_statistics(regressor, X_estim, y_estim)
print('R-squared:', r2)
print('Mean squared error:', mse)
print('Sigma:', sigma)

# Battery capacity is twelve times the standard deviation of the prediction error
# Because we want to have six sigma on both sides from the middle so that there
# is only very low chance that we exceed the capacity
DEMAND = 1.0
CAPACITY = 12 * sigma
T = len(y_sim)

turbine_pred = np.zeros(T)
turbine_true = np.zeros(T)
gas_power = np.zeros(T)
target = np.zeros(T)
energy = np.zeros(T + 1)
energy[0] = CAPACITY / 2

for t in range(T):
    turbine_pred[t] = regressor.predict(X_sim[t:t+1])[0]
    target[t] = DEMAND + (CAPACITY / 2 - energy[t])
    gas_power[t] = max(target[t] - turbine_pred[t], 0)
    turbine_true[t] = y_sim[t]
    energy[t + 1] = energy[t] + gas_power[t] + turbine_true[t] - DEMAND

plt.plot(energy, 'k')
plt.plot(np.zeros(T + 1), 'r')
plt.plot(np.ones(T + 1) * CAPACITY, 'r')
plt.show()
