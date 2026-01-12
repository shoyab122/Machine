import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt  # <-- Needed!

# Sample data
data = {
 'Height': [170, 165, 180, 175, 160, 172, 168, 177, 162, 158],
 'Weight': [65, 59, 75, 68, 55, 70, 62, 74, 58, 54],
 'Age': [30, 25, 35, 28, 22, 32, 27, 33, 24, 21],
 'Gender': [1, 0, 1, 1, 0, 1, 0, 1, 0, 0]
}

df = pd.DataFrame(data)

# Split features and target
X = df.drop('Gender', axis=1)
y = df['Gender']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Logistic Regression
model = LogisticRegression()
model.fit(X_pca, y)

# Confusion matrix
y_pred = model.predict(X_pca)
cm = confusion_matrix(y, y_pred)
print("Confusion Matrix:\n", cm)

# Plot PCA
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='bwr', edgecolors='k')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA of Height, Weight, Age (Gender)')
plt.show()
