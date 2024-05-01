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

