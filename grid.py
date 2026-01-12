from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.datasets import make_classification

# Generate synthetic dataset
x, y = make_classification(
    n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=42
)

# Define parameter space
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}

# Logistic Regression model
logreg = LogisticRegression(max_iter=10000)

# GridSearchCV
logreg_cv = GridSearchCV(logreg, param_grid, cv=5)

# Fit model
logreg_cv.fit(x, y)

# Print results
print("Tuned Logistic Regression Parameters: {}".format(logreg_cv.best_params_))
print("Best score is {}".format(logreg_cv.best_score_))