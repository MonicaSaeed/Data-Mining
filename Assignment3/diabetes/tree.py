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




# read data from diabetes_prediction_dataset.csv
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tree import build_tree
import numpy as np

df = pd.read_csv('diabetes_prediction_dataset.csv')
X = df.drop(columns=['diabetes'])
y = df['diabetes'].values

# Convert categorical variables to numerical using LabelEncoder
categorical_features = X.select_dtypes(include=[np.object_]).columns.to_list()
categorical_encoders = {}
for feature in categorical_features:
    categorical_encoders[feature] = LabelEncoder()
    X[feature] = categorical_encoders[feature].fit_transform(X[feature])

# print categorical_encoders
for feature in categorical_features:
    print(f'{feature}: {dict(zip(categorical_encoders[feature].classes_, categorical_encoders[feature].transform(categorical_encoders[feature].classes_)))}')    

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Normalize numerical features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# print numerical features after scaling

# print(X_train_scaled[:5])
# print(y_train[:5])
for i in range(5):
    print(f"Data record {i + 1}: {X_train_scaled[i]} - Actual: {y_train[i]}")

# Build decision tree
tree = build_tree(X_train_scaled, y_train)

# Define the Tree class and build_tree function

# Print tree structure
# def print_tree(node, depth=0):
#     if node is None:
#         return
#     print("  " * depth + f"X{node.feature_index} < {node.threshold} [Gini: {node.gini:.3f}, Samples: {node.num_samples}, Class Distribution: {node.num_samples_per_class}, Predicted Class: {node.predicted_class}]")
#     print_tree(node.left, depth + 1)
#     print_tree(node.right, depth + 1)

# Call the print_tree function to print the tree structure
# print_tree(tree)

# Predict using decision tree model
def predict_sample(tree, x):
    while tree.left:
        if x[tree.feature_index] < tree.threshold:
            tree = tree.left
        else:
            tree = tree.right
    return tree.predicted_class

def predict(X, tree):
    return [predict_sample(tree, x) for x in X]

y_pred = predict(X_test_scaled, tree)

# Evaluate decision tree model
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

a = accuracy(y_test, y_pred)
print(f"Accuracy: {a}")

for i in range(10):
    print(f"Data record {i + 1}: {X_test.iloc[i].values} - Actual: {y_test[i]}, Predicted: {y_pred[i]}")


# # get gender,age,hypertension,heart_disease,smoking_history,bmi,HbA1c_level,blood_glucose_level from user and predict diabetes 
# g = input("Enter gender: ")
# a = input("Enter age: ")
# h = input("Enter hypertension: ")
# hd = input("Enter heart disease: ")
# sh = input("Enter smoking history: ")
# bmi = input("Enter bmi: ")
# hba1c = input("Enter HbA1c level: ")
# bgl = input("Enter blood glucose level: ")


# # Convert categorical with fit transform
# g= categorical_encoders["gender"].transform([g])
# sh = categorical_encoders["smoking_history"].transform([sh])
# print(g, sh)

# # Normalize numerical features using StandardScaler
# data = [[g, a, h, hd, sh, bmi, hba1c, bgl]]
# data_scaled = scaler.transform(data)
# print(data_scaled)

# # Predict diabetes
# prediction = predict(data_scaled, tree)
# print(f"Predicted diabetes: {prediction[0]}")

