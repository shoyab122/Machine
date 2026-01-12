from sklearn.datasets import load_digits
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt
digits = load_digits()
isomap = Isomap(n_neighbors=30, n_components=2)
digits_isomap = isomap.fit_transform(digits.data)
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].scatter(digits.data[:, 0], digits.data[:, 1], c=digits.target, cmap=plt.cm.tab10)
ax[0].set_title('Original 2D Data (First Two Features)')
ax[1].scatter(digits_isomap[:, 0], digits_isomap[:, 1], c=digits.target, cmap=plt.cm.tab10)
ax[1].set_title('Isomap Reduced 2D Data')
plt.show()