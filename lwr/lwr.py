# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

print("Program started...")

# Load dataset
data = fetch_california_housing()

# Select one feature
X = data.data[:, 0]   # Median income
Y = data.target       # House price

print("Dataset loaded")

# Use only first 500 records (LWR is slow for very large data)
X = X[:500]
Y = Y[:500]

print("Using 500 records for training")

# Add intercept column
X_mat = np.c_[np.ones(len(X)), X]

# Bandwidth parameter
tau = 0.5


# Locally Weighted Regression Function
def locally_weighted_regression(x_query, X, Y, tau):

    m = X.shape[0]

    # Weight matrix
    W = np.eye(m)

    for i in range(m):
        diff = x_query - X[i, 1]
        W[i, i] = np.exp(-(diff ** 2) / (2 * tau ** 2))

    # Calculate theta
    theta = np.linalg.pinv(X.T @ W @ X) @ (X.T @ W @ Y)

    # Prediction
    y_pred = np.array([1, x_query]) @ theta

    return y_pred


print("Calculating predictions...")

# Predict values
y_pred_list = []

for x in X:
    y_pred = locally_weighted_regression(x, X_mat, Y, tau)
    y_pred_list.append(y_pred)

print("Prediction completed")


# Sort values for smooth curve
sorted_index = np.argsort(X)
X_sorted = X[sorted_index]
Y_sorted = Y[sorted_index]
y_pred_sorted = np.array(y_pred_list)[sorted_index]


print("Displaying graph...")

# Plot graph
plt.figure(figsize=(8,6))
plt.scatter(X_sorted, Y_sorted)
plt.plot(X_sorted, y_pred_sorted)

plt.xlabel("Median Income")
plt.ylabel("House Price")
plt.title("Locally Weighted Regression using California Housing Dataset")

plt.show()

input("Press Enter to exit...")