import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Define the directory for saving models
model_path = r"C:\Users\Dell\Desktop\heart project"
os.makedirs(model_path, exist_ok=True)  # Ensure the directory exists

# 🔹 Load your dataset (Replace with actual CSV file)
data_path = os.path.join(model_path, r"C:\Users\Dell\Desktop\heart project\cleaned_data.csv")
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
else:
    print(f"⚠️ Error: {data_path} not found! Please provide the correct dataset path.")
    exit()

# 🔹 Define features (X) and target variable (y)
X = df.drop(columns=['target'])  # Replace 'target' with the actual column name
y = df['target']  # Replace 'target' with the actual label column

# 🔹 Split the data into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Load trained models
dt_model = joblib.load(r"C:\Users\Dell\Desktop\heart project\decision_tree_model.pkl")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
nb_model = GaussianNB().fit(X_train, y_train)
svm_model = SVC(kernel='linear', random_state=42).fit(X_train, y_train)

# Save models
joblib.dump(rf_model, os.path.join(model_path, "random_forest_model.pkl"))
joblib.dump(nb_model, os.path.join(model_path, "naive_bayes_model.pkl"))
joblib.dump(svm_model, os.path.join(model_path, "svm_model.pkl"))

print("✅ Models have been successfully saved!")


# Streamlit app title
st.title("💖 Heart Disease Prediction App")
st.write("Enter your health details below to assess the risk of heart disease.")

# Sidebar for user input
st.sidebar.header("Patient Information")
age = st.sidebar.slider("Age", 20, 100, 50)
bmi = st.sidebar.slider("BMI", 10, 50, 25)
chol = st.sidebar.slider("Cholesterol Level", 100, 400, 200)
trestbps = st.sidebar.slider("Blood Pressure", 80, 200, 120)
exang = st.sidebar.radio("Exercise-Induced Angina (1 = Yes, 0 = No)", [0, 1])



# Convert input to DataFrame
patient_df = pd.DataFrame([[age, bmi, chol, trestbps, exang]], 
                          columns=['age', 'bmi', 'chol', 'trestbps', 'exang'])

# Load cleaned dataset for feature alignment
df_cleaned = pd.read_csv(r"C:\Users\Dell\Desktop\heart project\cleaned_data.csv")
expected_features = dt_model.feature_names_in_
patient_df = patient_df.reindex(columns=expected_features, fill_value=0)


# Load trained models (make sure these .pkl files exist)
rf_model = joblib.load(r"C:\\Users\\Dell\\Desktop\\heart project\\random_forest_model.pkl")
nb_model = joblib.load(r"C:\\Users\\Dell\\Desktop\\heart project\\naive_bayes_model.pkl")
svm_model = joblib.load(r"C:\\Users\\Dell\\Desktop\\heart project\\svm_model.pkl")


# Predict using ML models
dt_prediction = dt_model.predict(patient_df)[0]
rf_prediction = rf_model.predict(patient_df)[0]  # No need to fit
nb_prediction = nb_model.predict(patient_df)[0]  # No need to fit
svm_prediction = svm_model.predict(patient_df)[0]  # No need to fit


# Define Fuzzy Logic System
bmi_fuzzy = ctrl.Antecedent(np.arange(10, 50, 1), 'bmi')
age_fuzzy = ctrl.Antecedent(np.arange(20, 100, 1), 'age')
chol_fuzzy = ctrl.Antecedent(np.arange(100, 400, 1), 'chol')
exang_fuzzy = ctrl.Antecedent(np.arange(0, 2, 1), 'exang')
risk_fuzzy = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'risk')

bmi_fuzzy['low'] = fuzz.trimf(bmi_fuzzy.universe, [10, 18, 25])
bmi_fuzzy['medium'] = fuzz.trimf(bmi_fuzzy.universe, [20, 30, 40])
bmi_fuzzy['high'] = fuzz.trimf(bmi_fuzzy.universe, [35, 45, 50])

age_fuzzy['young'] = fuzz.trimf(age_fuzzy.universe, [20, 30, 40])
age_fuzzy['middle'] = fuzz.trimf(age_fuzzy.universe, [35, 50, 70])
age_fuzzy['old'] = fuzz.trimf(age_fuzzy.universe, [60, 80, 100])

chol_fuzzy['low'] = fuzz.trimf(chol_fuzzy.universe, [100, 150, 200])
chol_fuzzy['medium'] = fuzz.trimf(chol_fuzzy.universe, [180, 220, 260])
chol_fuzzy['high'] = fuzz.trimf(chol_fuzzy.universe, [240, 300, 400])

exang_fuzzy['no'] = fuzz.trimf(exang_fuzzy.universe, [0, 0, 1])
exang_fuzzy['yes'] = fuzz.trimf(exang_fuzzy.universe, [1, 1, 1])

risk_fuzzy['low'] = fuzz.trimf(risk_fuzzy.universe, [0, 0.2, 0.4])
risk_fuzzy['medium'] = fuzz.trimf(risk_fuzzy.universe, [0.3, 0.5, 0.7])
risk_fuzzy['high'] = fuzz.trimf(risk_fuzzy.universe, [0.6, 0.8, 1])

rule1 = ctrl.Rule(bmi_fuzzy['high'] & age_fuzzy['old'] & chol_fuzzy['high'] & exang_fuzzy['yes'], risk_fuzzy['high'])
rule2 = ctrl.Rule(bmi_fuzzy['medium'] & age_fuzzy['middle'] & chol_fuzzy['medium'] & exang_fuzzy['no'], risk_fuzzy['medium'])
rule3 = ctrl.Rule(bmi_fuzzy['low'] & age_fuzzy['young'] & chol_fuzzy['low'] & exang_fuzzy['no'], risk_fuzzy['low'])

risk_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
risk_sim = ctrl.ControlSystemSimulation(risk_ctrl)

if np.isnan(bmi) or np.isnan(age) or np.isnan(chol) or np.isnan(exang):
    print("⚠️ Error: One or more inputs are NaN!")

risk_sim.input['bmi'] = bmi
risk_sim.input['age'] = age
risk_sim.input['chol'] = chol
risk_sim.input['exang'] = exang
risk_sim.compute()

fuzzy_risk_score = risk_sim.output['risk']
fuzzy_risk_score = risk_sim.output.get('risk', None)  # Avoid KeyError

# Display Predictions
st.subheader("🩺 Prediction Results")
st.write(f"🔹 **Decision Tree Prediction:** {'High Risk' if dt_prediction else 'Low Risk'}")
st.write(f"🔹 **Random Forest Prediction:** {'High Risk' if rf_prediction else 'Low Risk'}")
st.write(f"🔹 **Naïve Bayes Prediction:** {'High Risk' if nb_prediction else 'Low Risk'}")
st.write(f"🔹 **SVM Prediction:** {'High Risk' if svm_prediction else 'Low Risk'}")

# Display Fuzzy Logic Risk Score
st.subheader("🔎 Fuzzy Logic Risk Score")
st.write(f"Your calculated risk score: **{fuzzy_risk_score:.2f}**")

# Risk Level
if fuzzy_risk_score > 0.6:
    st.error("⚠️ You are at HIGH risk of heart disease. Please consult a doctor.")
elif 0.3 < fuzzy_risk_score <= 0.6:
    st.warning("⚠️ You are at MEDIUM risk. Consider making lifestyle changes.")
else:
    st.success("✅ You are at LOW risk. Keep maintaining a healthy lifestyle!")

# Visualizing Risk Score
st.subheader("📊 Risk Score Distribution")
st.bar_chart([fuzzy_risk_score])

# Footer
st.markdown("---")
st.write("💡 **Note:** This is a predictive model and does not replace professional medical advice. Consult a doctor for an accurate diagnosis.")
risk_sim.compute()  # Compute fuzzy output
print("🔍 Available keys in risk_sim.output:", risk_sim.output.keys())

if np.isnan(bmi) or np.isnan(age) or np.isnan(chol) or np.isnan(exang):
    print("⚠️ Error: One or more inputs are NaN!")

exang_fuzzy = ctrl.Antecedent(np.arange(0, 2, 1), 'exang')


risk_fuzzy = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'risk')

risk_fuzzy['low'] = fuzz.trimf(risk_fuzzy.universe, [0, 0.2, 0.4])
risk_fuzzy['medium'] = fuzz.trimf(risk_fuzzy.universe, [0.3, 0.5, 0.7])
risk_fuzzy['high'] = fuzz.trimf(risk_fuzzy.universe, [0.6, 0.8, 1])


if 'bmi' not in risk_sim.input:
    print("⚠️ Error: 'bmi' is not a recognized input for risk_sim!")