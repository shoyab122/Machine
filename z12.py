import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
# 1. Sample data
data = {
 'Height': [170, 165, 180, 175, 160, 172, 168, 177, 162, 158],
 'Weight': [65, 59, 75, 68, 55, 70, 62, 74, 58, 54],
 'Age': [30, 25, 35, 28, 22, 32, 27, 33, 24, 21],
 'Gender': [1, 0, 1, 1, 0, 1, 0, 1, 0, 0]
}
df = pd.DataFrame(data)

# 2. Features & target
X = df.drop('Gender', axis=1)
y = df['Gender']

# 3. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, y, test_size=0.3, random_state=42
)

# 6. Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 7. Evaluate
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
print("Confusion Matrix:\n", cm)
print(f"Accuracy: {acc*100:.2f}%")

# 8. Plot PCA components
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='bwr', edgecolors='k')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA of Height, Weight, Age (Gender)')
plt.show()
