import numpy as np

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # The index of the feature column in the dataset
        self.threshold = threshold  # The value of the feature used to split the data
        self.left = left  
        self.right = right  
        self.value = value  # Class label if node is a leaf

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.n_features = X.shape[1]
        self.tree = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_samples_per_class = [np.sum(y == i) for i in range(self.n_classes)]
        # If all samples belong to the same class or reached maximum depth
        if (len(np.unique(y)) == 1) or (depth == self.max_depth):
            return Node(value=np.argmax(n_samples_per_class))

        # Find the best split
        best_gini = float('inf')
        best_feature, best_threshold = None, None
        # Iterate over all features
        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = np.where(X[:, feature_index] <= threshold)[0]
                right_indices = np.where(X[:, feature_index] > threshold)[0]
                gini = self._gini_impurity(y[left_indices], y[right_indices])
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_index
                    best_threshold = threshold

        # Perform the split with the best found parameters
        left_indices = np.where(X[:, best_feature] <= best_threshold)[0]
        right_indices = np.where(X[:, best_feature] > best_threshold)[0]

        left_subtree = self._grow_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._grow_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(feature_index=best_feature, threshold=best_threshold,
                    left=left_subtree, right=right_subtree)

    def _gini_impurity(self, left_labels, right_labels):
        p_left = len(left_labels) / (len(left_labels) + len(right_labels))
        p_right = len(right_labels) / (len(left_labels) + len(right_labels))
        
        gini_left = 0 if len(left_labels) == 0 else 1 - sum([(np.sum(left_labels == c) / len(left_labels)) ** 2 for c in range(self.n_classes)])
        gini_right = 0 if len(right_labels) == 0 else 1 - sum([(np.sum(right_labels == c) / len(right_labels)) ** 2 for c in range(self.n_classes)])
        
        return p_left * gini_left + p_right * gini_right


    def predict(self, X):
        return [self._predict(x, self.tree) for x in X]

    def _predict(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict(x, node.left)
        return self._predict(x, node.right)
    
    def accuracy(self, y_true, y_pred):
        return np.sum(y_true == y_pred) / len(y_true)


