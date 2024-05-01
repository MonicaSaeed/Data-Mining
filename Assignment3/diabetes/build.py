from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# Load the data
data = pd.read_csv('diabetes_prediction_dataset.csv')

# Preprocessing the data
# Convert categorical variables to numerical values
data['gender'] = data['gender'].map({'Female': 0, 'Male': 1})
data['smoking_history'] = data['smoking_history'].map({'never': 0, 'former': 1, 'current': 2, 'No Info': None})

# Handling missing values
data.fillna(method='ffill', inplace=True)  # Forward fill missing values

# Separate numerical and categorical columns
numerical_columns = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
categorical_columns = ['gender', 'hypertension', 'heart_disease', 'smoking_history']

# Normalize numerical columns
scaler = MinMaxScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Splitting the data
X = data.drop(columns=['diabetes'])
y = data['diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Training the Decision Tree
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

for i in range(5):
    print(f"Data record {i + 1}: {X_test.iloc[i].values} - Actual: {y_test.iloc[i]} - Predicted: {y_pred[i]}")

# Calculating accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
