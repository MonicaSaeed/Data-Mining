import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from dt import DecisionTree
import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names, but MinMaxScaler was fitted with feature names")

def read(file_path, percentage):
    # Load the data
    num_rows = sum(1 for line in open(file_path)) 
    num_read_rows = int(num_rows * percentage / 100)
    data = pd.read_csv(file_path, nrows=num_read_rows)

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
    return X_train, X_test, y_train, y_test, categorical_encoders, scaler

def train(X_train, y_train):
    # Training the Decision Tree
    model = DecisionTree(max_depth=5)
    model.fit(X_train.values, y_train.values)  # Ensure you pass values instead of DataFrame
    return model


def test(model, X_test, y_test):
    # Making predictions
    y_pred = model.predict(X_test.values)  # Ensure you pass values instead of DataFrame

    # print predictions and print actual values and predictions
    # for i in range(5):
    #     print(f"Data record {i + 1}: {X_test.iloc[i].values} - Actual: {y_test.iloc[i]} - Predicted: {y_pred[i]}")

    # Calculating accuracy
    accuracy = model.accuracy(y_test.values, y_pred)  # Ensure you pass values instead of DataFrame
    # print("Accuracy:", accuracy)
    return y_pred, accuracy

def predict_row(model, categorical_encoders, scaler,row):
    # Encode categorical features
    g_encoded = categorical_encoders["gender"].transform([row[0]])[0]
    sh_encoded = categorical_encoders["smoking_history"].transform([row[4]])[0]
    # Normalize numerical input values
    sc = scaler.transform([[row[1], row[5], row[6], row[7]]])
    a, bmi, hba1c, bg = sc[0]

    # Make the prediction
    prediction = model.predict([[g_encoded, a, row[2], row[3], sh_encoded, bmi, hba1c, bg]])
    # print("Prediction:", prediction[0])
    return prediction[0]



file_path = 'diabetes_prediction_dataset.csv'
percentage = 100
X_train, X_test, y_train, y_test, categorical_encoders, scaler = read(file_path, percentage)
model = train(X_train, y_train)
y_pred, accuracy = test(model, X_test, y_test)
for i in range(5):
    print(f"Data record {i + 1}: {X_test.iloc[i].values} - Actual: {y_test.iloc[i]} - Predicted: {y_pred[i]}")
print("Accuracy:", accuracy)
# Female,79.0,0,0,No Info,23.86,5.7,85,0

# Predicting a single instance
g = "Female"
a = 79.0
h = 0
hd = 0
sh = "No Info"
bmi = 23.86
hba1c = 5.7
bg = 85
row=[g,a,h,hd,sh,bmi,hba1c,bg]
prediction = predict_row(model, categorical_encoders, scaler,row)
print(prediction)
