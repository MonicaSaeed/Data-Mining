import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split

from dt import DecisionTree

def read_data(file_path, percentage):
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

    print("Categorical Encoders: ")
    # print categorical_encoders map
    for feature in categorical_features:
        label_encoder = categorical_encoders[feature]
        mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        print(f'{feature}: {mapping}')

    # Discretize numerical columns into intervals
    numerical_columns = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    interval_encoders = {}
    for feature in numerical_columns:
        X[feature], interval_encoders[feature] = pd.cut(X[feature], bins=5, labels=False, retbins=True)

    print("\nInterval Encoders: ")
    # print interval_encoders map
    for feature in numerical_columns:
        label_encoder = interval_encoders[feature]
        mapping = dict(zip(label_encoder, range(len(label_encoder))))
        print(f'{feature}: {mapping}')

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test, categorical_encoders, interval_encoders

def traindt(X_train, y_train):
    # Training the Decision Tree
    model = DecisionTree(max_depth=8)
    model.fit(X_train.values, y_train.values)  # Ensure you pass values instead of DataFrame
    return model

def testdt(model, X_test, y_test):
    # Making predictions
    y_pred = model.predict(X_test.values)  # Ensure you pass values instead of DataFrame

    # print predictions and print actual values and predictions
    # for i in range(5):
    #     print(f"Data record {i + 1}: {X_test.iloc[i].values} - Actual: {y_test.iloc[i]} - Predicted: {y_pred[i]}")

    # Calculating accuracy
    accuracy = model.accuracy(y_test.values, y_pred)  # Ensure you pass values instead of DataFrame
    # print("Accuracy:", accuracy)
    return y_pred, accuracy

def predict_rowdt(model, categorical_encoders, interval_encoders, row):
    # Encode categorical features
    g_encoded = categorical_encoders["gender"].transform([row[0]])[0]
    sh_encoded = categorical_encoders["smoking_history"].transform([row[4]])[0]
    
    # Discretize numerical input values using the stored intervals
    a_bin = interval_encoders['age'].searchsorted(row[1]) - 1
    bmi_bin = interval_encoders['bmi'].searchsorted(row[5]) - 1
    hba1c_bin = interval_encoders['HbA1c_level'].searchsorted(row[6]) - 1
    bg_bin = interval_encoders['blood_glucose_level'].searchsorted(row[7]) - 1

    # Make the prediction
    prediction = model.predict([[g_encoded, a_bin, row[2], row[3], sh_encoded, bmi_bin, hba1c_bin, bg_bin]])
    return prediction[0]

def applydt(file_path, percentage):
    model = traindt(X_train, y_train)
    y_pred, accuracy = testdt(model, X_test, y_test)
    return X_train, X_test, y_train, y_test, categorical_encoders, interval_encoders, model, y_pred, accuracy


file_path = 'diabetes_prediction_dataset.csv'
percentage = 3
X_train, X_test, y_train, y_test, categorical_encoders, interval_encoders = read_data(file_path, percentage)

X_train, X_test, y_train, y_test, categorical_encoders, interval_encoders, model, y_pred, accuracy = applydt(file_path, percentage)
# y_pred length 
l=len(y_pred)
print("Data record: ",)
for i in range(l):
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
row = [g, a, h, hd, sh, bmi, hba1c, bg]
prediction = predict_rowdt(model, categorical_encoders, interval_encoders, row)
print("prediction: ",prediction)
