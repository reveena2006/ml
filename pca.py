import numpy as np
import pandas as pd

data = {
    'Marks': [70,65,80,75,60,85,78,68,90,72],
    'Attendance': [75,70,85,80,65,90,82,72,95,78],
    'Study_Hours': [6,5,8,7,4,9,7,6,10,6]
}

df = pd.DataFrame(data)
print("Original Dataset:\n")
print(df)

mean_vector = df.mean()
print("\nMean Vector:\n")
print(mean_vector)

X_centered = df - mean_vector
print("\nMean Centered Data:\n")
print(X_centered)

cov_matrix = np.cov(X_centered.T)
print("\nCovariance Matrix:\n")
print(cov_matrix)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("\nEigenvalues:\n")
print(eigenvalues)

print("\nEigenvectors:\n")
print(eigenvectors)

sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues_sorted = eigenvalues[sorted_indices]
eigenvectors_sorted = eigenvectors[:, sorted_indices]

print("\nSorted Eigenvalues:\n")
print(eigenvalues_sorted)

print("\nSorted Eigenvectors:\n")
print(eigenvectors_sorted)

explained_variance_ratio = eigenvalues_sorted / np.sum(eigenvalues_sorted)
print("\nExplained Variance Ratio:\n")
print(explained_variance_ratio)

k = 2
W = eigenvectors_sorted[:, :k]
print("\nProjection Matrix (Top 2 Eigenvectors):\n")
print(W)

# ✅ FIXED LINE
Z = np.dot(X_centered, W)

pca_df = pd.DataFrame(Z, columns=['PC1', 'PC2'])
print("\nData after PCA (Reduced to 2 Dimensions):\n")
print(pca_df)
