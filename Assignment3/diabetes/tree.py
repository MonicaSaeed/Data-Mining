import numpy as np
from sklearn.calibration import LabelEncoder

class Tree:
    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

def build_tree(X, y, depth=0):
    num_samples_per_class = [np.sum(y == i) for i in range(2)]
    predicted_class = np.argmax(num_samples_per_class)
    node = Tree(
        gini=1 - sum((np.sum(y == c) / y.size) ** 2 for c in range(2)),
        num_samples=y.size,
        num_samples_per_class=num_samples_per_class,
        predicted_class=predicted_class,
    )

    if depth < 2:
        idx, thr = best_split(X, y)
        if idx is not None:
            indices_left = X[:, idx] < thr
            X_left, y_left = X[indices_left], y[indices_left]
            X_right, y_right = X[~indices_left], y[~indices_left]
            node.feature_index = idx
            node.threshold = thr
            node.left = build_tree(X_left, y_left, depth + 1)
            node.right = build_tree(X_right, y_right, depth + 1)
    return node

def best_split(X, y):
    m, n = X.shape
    if m <= 1:
        return None, None

    num_parent = [np.sum(y == c) for c in range(2)]
    best_gini = 1 - sum((np.sum(y == c) / y.size) ** 2 for c in range(2))
    best_idx, best_thr = None, None

    for idx in range(n):
        thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
        num_left = [0] * 2
        num_right = num_parent.copy()
        for i in range(1, m):
            c = classes[i - 1]
            num_left[c] += 1
            num_right[c] -= 1
            gini_left = 1 - sum(
                (num_left[x] / i) ** 2 for x in range(2)
            )
            gini_right = 1 - sum(
                (num_right[x] / (m - i)) ** 2 for x in range(2)
            )
            gini = (i * gini_left + (m - i) * gini_right) / m
            if thresholds[i] == thresholds[i - 1]:
                continue
            if gini < best_gini:
                best_gini = gini
                best_idx = idx
                best_thr = (thresholds[i] + thresholds[i - 1]) / 2
    return best_idx, best_thr
