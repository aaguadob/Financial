import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNet, LinearRegression
import xgboost as xgb

import openbb as obb
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch
from models import DNN
import torch.optim
import torch.nn
from sklearn.preprocessing import StandardScaler

output = obb.obb.equity.price.historical("AAPL", start_date="2014-01-01", end_date="2024-02-28")
df = output.to_dataframe()

x_open = []
x_close = []
y = []
n= 5
for i in range(len(df)-n):
    x_open.append(np.array(df["open"].iloc[i:i+n]))
    x_close.append(np.array(df["close"].iloc[i:i+n]))
    y.append(df["open"].iloc[i+n])
x_open = [arr.flatten() for arr in x_open]
x_close = [arr.flatten() for arr in x_close]

df_2 = pd.DataFrame(x_open)
df_3 = pd.DataFrame(y, columns=["label"])
df_comb = pd.concat([df_2, df_3], axis=1)
weights = np.abs(df_comb["label"] - df_comb[n-1])
df_comb["label"] = np.where(df_comb["label"] > df_comb[n-1], 1, 0)
weights = torch.tensor(weights)
columns = df_comb.columns.to_list()
y = df_comb["label"]
columns.remove("label")
X = df_comb[columns]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.99,random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
architecture = [20, 20]
modell = DNN(architecture = architecture, input_size=X_train.shape[1])

epochs = 5000
# Add a small epsilon value to avoid issues in the loss calculation
import torch.nn.functional as F

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

# Apply the weight initialization
modell.apply(init_weights)

# Set learning rate and optimizer
lr = 1e-5  # Try a smaller learning rate
optimizer = torch.optim.Adam(modell.parameters(), lr=lr)
loss = torch.nn.BCEWithLogitsLoss()

for epoch in range(epochs):
    print(f"Epoch {epoch}")
    modell.train()

    # Forward pass
    y_pred = modell(X_train)

    # Ensure numerical stability before calculating the loss
    epsilon = 1e-8
    y_pred = torch.clamp(y_pred, min=epsilon, max=1 - epsilon)

    l = loss(y_train, y_pred)

    # Check if loss is NaN
    if torch.isnan(l):
        print("NaN detected in loss")
        break

    print(f"Loss: {l.item()}")

    # Backward pass
    l.backward()

    # Gradient clipping (limits the max gradient norm)
    torch.nn.utils.clip_grad_norm_(modell.parameters(), max_norm=1.0)

    # Check for NaNs or large gradients
    for name, param in modell.named_parameters():
        if torch.any(torch.isnan(param.grad)) or torch.any(torch.isinf(param.grad)):
            print(f"NaNs or Infs detected in gradients for parameter: {name}")
    
    # Check gradient norms
    for name, param in modell.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm()
            print(f"Gradient norm for {name}: {grad_norm}")

    # Optimizer step
    optimizer.step()

    # Zero gradients
    optimizer.zero_grad()




