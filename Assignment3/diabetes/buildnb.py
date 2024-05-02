import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data from CSV file
data = pd.read_csv('diabetes_prediction_dataset.csv')
# Convert categorical variables to numerical values
data['gender'] = data['gender'].map({'Female': 0, 'Male': 1})
data['smoking_history'] = data['smoking_history'].map({'never': 0, 'former': 1, 'current': 2, 'No Info': None})

# Handling missing values
data.fillna(method='ffill', inplace=True)  # Forward fill missing values



# Impute missing values with median
data.fillna(data.median(), inplace=True)

# Encode categorical variables
label_encoder = LabelEncoder()
data['gender'] = label_encoder.fit_transform(data['gender'])
data['smoking_history'] = label_encoder.fit_transform(data['smoking_history'])

# Split data into features and target variable
X = data.drop(columns=["diabetes"])
y = data["diabetes"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = GaussianNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
