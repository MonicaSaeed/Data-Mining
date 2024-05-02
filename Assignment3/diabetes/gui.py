import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.model_selection import train_test_split
from dt import DecisionTree
from nb import NaiveBayes

class App:
    def __init__(self, master):
        self.master = master
        master.title("Diabetes Prediction")
        master.geometry("1000x700")

        # Labels and Entries
        self.label_file = tk.Label(master, text="Select File:")
        self.label_file.grid(row=0, column=0, padx=10, pady=10, sticky="w")
        
        self.entry_file = tk.Entry(master, width=50)
        self.entry_file.grid(row=0, column=1, padx=10, pady=10)

        self.label_percentage = tk.Label(master, text="Percentage of Data to Use:")
        self.label_percentage.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        self.entry_percentage = tk.Entry(master)
        self.entry_percentage.grid(row=1, column=1, padx=10, pady=10)

        # Buttons
        self.button_browse = tk.Button(master, text="Browse", command=self.browse_file)
        self.button_browse.grid(row=0, column=2, padx=10, pady=10)

        self.button_process = tk.Button(master, text="Process Data", command=self.process_data)
        self.button_process.grid(row=2, column=1, padx=10, pady=10)

        self.button_predict_dt = tk.Button(master, text="Predict DT", command=self.predict_dt)
        self.button_predict_dt.grid(row=2, column=0, padx=10, pady=10)

        self.button_predict_nb = tk.Button(master, text="Predict NB", command=self.predict_nb)
        self.button_predict_nb.grid(row=2, column=2, padx=10, pady=10)

        # Text Widgets for displaying results
        self.result_text_dt = scrolledtext.ScrolledText(master, wrap=tk.WORD, width=50, height=30)
        self.result_text_dt.grid(row=3, column=0, padx=10, pady=10)

        self.result_text_nb = scrolledtext.ScrolledText(master, wrap=tk.WORD, width=50, height=30)
        self.result_text_nb.grid(row=3, column=1, padx=10, pady=10)

        # Input fields for prediction
        self.gender_var = tk.StringVar()
        self.age_var = tk.StringVar()
        self.hypertension_var = tk.StringVar()
        self.heart_disease_var = tk.StringVar()
        self.smoking_history_var = tk.StringVar()
        self.bmi_var = tk.StringVar()
        self.hba1c_level_var = tk.StringVar()
        self.blood_glucose_level_var = tk.StringVar()

        self.label_gender = tk.Label(master, text="Gender:")
        self.label_gender.grid(row=4, column=0, padx=10, pady=5)
        self.entry_gender = tk.Entry(master, textvariable=self.gender_var)
        self.entry_gender.grid(row=4, column=1, padx=10, pady=5)

        self.label_age = tk.Label(master, text="Age:")
        self.label_age.grid(row=5, column=0, padx=10, pady=5)
        self.entry_age = tk.Entry(master, textvariable=self.age_var)
        self.entry_age.grid(row=5, column=1, padx=10, pady=5)

        self.label_hypertension = tk.Label(master, text="Hypertension:")
        self.label_hypertension.grid(row=6, column=0, padx=10, pady=5)
        self.entry_hypertension = tk.Entry(master, textvariable=self.hypertension_var)
        self.entry_hypertension.grid(row=6, column=1, padx=10, pady=5)

        self.label_heart_disease = tk.Label(master, text="Heart Disease:")
        self.label_heart_disease.grid(row=7, column=0, padx=10, pady=5)
        self.entry_heart_disease = tk.Entry(master, textvariable=self.heart_disease_var)
        self.entry_heart_disease.grid(row=7, column=1, padx=10, pady=5)

        self.label_smoking_history = tk.Label(master, text="Smoking History:")
        self.label_smoking_history.grid(row=8, column=0, padx=10, pady=5)
        self.entry_smoking_history = tk.Entry(master, textvariable=self.smoking_history_var)
        self.entry_smoking_history.grid(row=8, column=1, padx=10, pady=5)

        self.label_bmi = tk.Label(master, text="BMI:")
        self.label_bmi.grid(row=9, column=0, padx=10, pady=5)
        self.entry_bmi = tk.Entry(master, textvariable=self.bmi_var)
        self.entry_bmi.grid(row=9, column=1, padx=10, pady=5)

        self.label_hba1c_level = tk.Label(master, text="HbA1c Level:")
        self.label_hba1c_level.grid(row=10, column=0, padx=10, pady=5)
        self.entry_hba1c_level = tk.Entry(master, textvariable=self.hba1c_level_var)
        self.entry_hba1c_level.grid(row=10, column=1, padx=10, pady=5)

        self.label_blood_glucose_level = tk.Label(master, text="Blood Glucose Level:")
        self.label_blood_glucose_level.grid(row=11, column=0, padx=10, pady=5)
        self.entry_blood_glucose_level = tk.Entry(master, textvariable=self.blood_glucose_level_var)
        self.entry_blood_glucose_level.grid(row=11, column=1, padx=10, pady=5)

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")])
        self.entry_file.delete(0, tk.END)
        self.entry_file.insert(0, file_path)

    def process_data(self):
        file_path = self.entry_file.get()
        percentage = int(self.entry_percentage.get())
        
        if not file_path:
            messagebox.showerror("Error", "Please select a file.")
            return

        try:
            X_train, X_test, y_train, y_test, categorical_encoders, interval_encoders = read_data(file_path, percentage)
            self.apply_decision_tree(X_train, y_train, X_test, y_test)
            self.apply_naive_bayes(X_train, y_train, X_test, y_test)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def apply_decision_tree(self, X_train, y_train, X_test, y_test):
        model = traindt(X_train, y_train)
        y_pred, accuracy = testdt(model, X_test, y_test)
        self.print_results("Decision Tree", X_test, y_test, y_pred, accuracy, self.result_text_dt)

    def apply_naive_bayes(self, X_train, y_train, X_test, y_test):
        model = trainnb(X_train, y_train)
        y_pred, accuracy = testnb(model, X_test, y_test)
        self.print_results("Naive Bayes", X_test, y_test, y_pred, accuracy, self.result_text_nb)

    def predict_dt(self):
        try:
            row = [
                self.gender_var.get(),
                float(self.age_var.get()),
                int(self.hypertension_var.get()),
                int(self.heart_disease_var.get()),
                self.smoking_history_var.get(),
                float(self.bmi_var.get()),
                float(self.hba1c_level_var.get()),
                float(self.blood_glucose_level_var.get())
            ]
            file_path = self.entry_file.get()
            percentage = int(self.entry_percentage.get())
            X_train, X_test, y_train, y_test, categorical_encoders, interval_encoders = read_data(file_path, percentage)
            model = traindt(X_train, y_train)
            prediction = predict_rowdt(model, categorical_encoders, interval_encoders, row)
            self.result_text_dt.insert(tk.END, f"Prediction (DT): {prediction}\n")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def predict_nb(self):
        try:
            row = [
                self.gender_var.get(),
                float(self.age_var.get()),
                int(self.hypertension_var.get()),
                int(self.heart_disease_var.get()),
                self.smoking_history_var.get(),
                float(self.bmi_var.get()),
                float(self.hba1c_level_var.get()),
                float(self.blood_glucose_level_var.get())
            ]
            file_path = self.entry_file.get()
            percentage = int(self.entry_percentage.get())
            X_train, X_test, y_train, y_test, categorical_encoders, interval_encoders = read_data(file_path, percentage)
            model = trainnb(X_train, y_train)
            prediction = predict_rownb(model, categorical_encoders, interval_encoders, row)
            self.result_text_nb.insert(tk.END, f"Prediction (NB): {prediction}\n")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

    def print_results(self, model_name, X_test, y_test, y_pred, accuracy, result_text):
        result_text.insert(tk.END, f"\n{model_name} Results:\n")
        result_text.insert(tk.END, f"Accuracy: {accuracy}\n")
        l = len(y_pred)
        for i in range(l):
            result_text.insert(tk.END, f"Data record {i + 1}: {X_test.iloc[i].values} - Actual: {y_test.iloc[i]} - Predicted: {y_pred[i]}\n")

def read_data(file_path, percentage):
    num_rows = sum(1 for line in open(file_path)) 
    num_read_rows = int(num_rows * percentage / 100)
    data = pd.read_csv(file_path, nrows=num_read_rows)

    # Find and handle missing values
    # print(data.isna().sum())
    data = data.dropna()

    X = data.drop(columns=['diabetes'])
    y = data['diabetes']

    categorical_features = X.select_dtypes(include=[np.object_]).columns.to_list()
    categorical_encoders = {}
    for feature in categorical_features:
        categorical_encoders[feature] = LabelEncoder()
        X[feature] = categorical_encoders[feature].fit_transform(X[feature])

    print("Categorical Encoders: ")
    for feature in categorical_features:
        label_encoder = categorical_encoders[feature]
        mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        print(f'{feature}: {mapping}')

    numerical_columns = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    interval_encoders = {}
    for feature in numerical_columns:
        X[feature], interval_encoders[feature] = pd.cut(X[feature], bins=5, labels=False, retbins=True)

    print("\nInterval Encoders: ")
    for feature in numerical_columns:
        label_encoder = interval_encoders[feature]
        mapping = dict(zip(label_encoder, range(len(label_encoder))))
        print(f'{feature}: {mapping}')

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    return X_train, X_test, y_train, y_test, categorical_encoders, interval_encoders

def traindt(X_train, y_train):
    model = DecisionTree(max_depth=8)
    model.fit(X_train.values, y_train.values)
    return model

def testdt(model, X_test, y_test):
    y_pred = model.predict(X_test.values)
    accuracy = model.accuracy(y_test.values, y_pred)
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

def trainnb(X_train, y_train):
    model = NaiveBayes()
    model.fit(X_train.values, y_train.values)
    return model

def testnb(model, X_test, y_test):
    y_pred = model.predict(X_test.values)
    accuracy = model.accuracy(y_test.values, y_pred)
    return y_pred, accuracy

def predict_rownb(model, categorical_encoders, interval_encoders, row):
    # Encode categorical features
    g_encoded = categorical_encoders["gender"].transform([row[0]])[0]
    sh_encoded = categorical_encoders["smoking_history"].transform([row[4]])[0]
    
    # Discretize numerical input values using the stored intervals
    a_bin = interval_encoders['age'].searchsorted(row[1]) - 1
    bmi_bin = interval_encoders['bmi'].searchsorted(row[5]) - 1
    hba1c_bin = interval_encoders['HbA1c_level'].searchsorted(row[6]) - 1
    bg_bin = interval_encoders['blood_glucose_level'].searchsorted(row[7]) - 1

    prediction = model.predict([[g_encoded, a_bin, row[2], row[3], sh_encoded, bmi_bin, hba1c_bin, bg_bin]])
    return prediction[0]

root = tk.Tk()
app = App(root)
root.mainloop()
