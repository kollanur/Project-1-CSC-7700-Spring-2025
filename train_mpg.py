from mlp import *
import os
import numpy as np
import matplotlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# URL for Auto MPG dataset from UCI
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"

# Fix column names by adding "car_name"
columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "model_year", "origin", "car_name"]

# Load dataset correctly
df = pd.read_csv(url, sep="\\s+", names=columns, na_values="?")  

# Drop 'car_name' since it's not needed for numerical analysis
df.drop(columns=["car_name"], inplace=True)

# Drop missing values
df.dropna(inplace=True)

# Ensure 'origin' is an integer
df["origin"] = df["origin"].astype(int)

# Convert categorical column ('origin') to numerical (one-hot encoding)
df = pd.get_dummies(df, columns=['origin'], drop_first=True)

# Separate features and target variable
X = df.drop(columns=["mpg"])
y = df["mpg"].values.reshape(-1, 1)

# Normalize numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training (70%), validation (15%), and testing (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

# Print dataset shapes
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

def create_deep_mlp(input_size=X_train.shape[1], hidden_units=128, num_layers=4, output_size=1):
    layers = []
    prev_units = input_size  # Input layer size
    
    for _ in range(num_layers - 1):  # Hidden Layers with ReLU
        layers.append(Layer(fan_in=prev_units, fan_out=hidden_units, activation_function=Relu()))
        prev_units = hidden_units

    # Output layer with Linear activation for regression
    layers.append(Layer(fan_in=prev_units, fan_out=output_size, activation_function=Linear()))

    return MultilayerPerceptron(layers)



# Create a deep MLP for regression
mlp = create_deep_mlp(num_layers=8)

# Define loss function for regression
loss_func = SquaredError()

# Train the model
training_losses, validation_losses = mlp.train(
    train_x=X_train, train_y=y_train,
    val_x=X_test, val_y=y_test,
    loss_func=loss_func,
    learning_rate=0.0001, batch_size=64, epochs=70,
    rmsprop=True,  # Enable RMSProp
    beta=0.9,      # RMSProp decay factor
    epsilon=1e-8   # Small constant for numerical stability
)


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Evaluate the model on the test set
y_pred = mlp.forward(X_test, training=False)

# Convert predictions to 1D array for regression evaluation
y_pred = y_pred.flatten()
y_test = y_test.flatten()

# Compute regression performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(training_losses, label="Training Loss", color="blue")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.legend()

# Plot validation loss
plt.subplot(1, 2, 2)
plt.plot(validation_losses, label="Validation Loss", color="red")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Validation Loss Over Time")
plt.legend()

plt.show()
