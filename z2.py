import numpy as np
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=3000, n_features=30, n_informative=25, n_classes=2, random_state=62)

from scipy.stats import randint
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV

param_dist = {"max_depth": [4, None],
              "max_features": randint(1, 15),
              "min_samples_leaf": randint(1, 15),
              "criterion": ["gini", "entropy"]}
tree = DecisionTreeClassifier()
tree_cv = RandomizedSearchCV(tree, param_dist, cv=8)
tree_cv.fit(X, y)
print("Tuned Decision Tree Parameters:{}",format(tree_cv.best_params_))
print("Best score:{}",format(tree_cv.best_score_))