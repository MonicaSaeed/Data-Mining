import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # Index of feature to split on
        self.threshold = threshold  # Threshold value for the feature
        self.left = left  # Left child (subtree)
        self.right = right  # Right child (subtree)
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
            # Return a leaf node
            return Node(value=np.argmax(n_samples_per_class))

        # Find the best split
        best_gini = float('inf')
        best_feature, best_threshold = None, None
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


# Example usage
# Load the data
data = pd.read_csv('diabetes_prediction_dataset.csv')

# Find and handle missing values
# print(data.isna().sum())
data = data.dropna()

# Preprocessing steps...
X = data.drop(columns=['diabetes'])
y = data['diabetes']

# Convert categorical features to numerical using LabelEncoder
categorical_features = X.select_dtypes(include=[np.object_]).columns.to_list()
categorical_encoders = {}
for feature in categorical_features:
    categorical_encoders[feature] = LabelEncoder()
    X[feature] = categorical_encoders[feature].fit_transform(X[feature])

# Normalize numerical columns
scaler = MinMaxScaler()
numerical_columns = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
X[numerical_columns] = scaler.fit_transform(X[numerical_columns])
# # drop numerical_columns
# X = X.drop(columns=numerical_columns)


# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Training the Decision Tree
model = DecisionTree(max_depth=5)
model.fit(X_train.values, y_train.values)  # Ensure you pass values instead of DataFrame

# Making predictions
y_pred = model.predict(X_test.values)  # Ensure you pass values instead of DataFrame

# print predictions and print actual values and predictions
for i in range(5):
    print(f"Data record {i + 1}: {X_test.iloc[i].values} - Actual: {y_test.iloc[i]} - Predicted: {y_pred[i]}")

# Calculating accuracy
accuracy = model.accuracy(y_test.values, y_pred)  # Ensure you pass values instead of DataFrame
print("Accuracy:", accuracy)

# Predicting a single instance
g="Female"
a=79.0
h=0
hd=0
sh="No Info"
bmi=23.86
hba1c=5.7
bg=85
# categorical encoding
g = categorical_encoders["gender"].transform([g])[0]
sh = categorical_encoders["smoking_history"].transform([sh])[0]
# Normalize numerical input values
sc = scaler.transform([[a, bmi, hba1c, bg]])
a, bmi, hba1c, bg = sc[0]

p = [[g, a, h, hd, sh, bmi, hba1c, bg]]
# Make the prediction
prediction = model.predict(p)
print("Prediction:", prediction[0])