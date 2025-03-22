import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import collections
if not hasattr(collections, 'Mapping'):
    collections.Mapping = collections.abc.Mapping
from experta import Fact, Field, KnowledgeEngine, Rule, P, AND
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.tree import DecisionTreeClassifier, plot_tree
import joblib
from sklearn.metrics import classification_report
import streamlit as st

# Load dataset
df = pd.read_csv(r"C:\Users\Dell\Desktop\heart project\heart (1).csv")

# Convert all column names to lowercase for consistency
df.columns = df.columns.str.lower()

# Handle missing values (fill numerical with mean, categorical with mode)
df.fillna(df.mean(numeric_only=True), inplace=True)
for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Normalize numerical features using MinMaxScaler
num_features = df.select_dtypes(include=['int64', 'float64']).columns
scaler = MinMaxScaler()
df[num_features] = scaler.fit_transform(df[num_features])

# Encode categorical variables using One-Hot Encoding
cat_features = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=cat_features, drop_first=True)

# Feature Selection using Correlation (keeping features with correlation > 0.1)
correlation = df.corr()['target'].abs().sort_values(ascending=False)
selected_features = correlation[correlation > 0.1].index.tolist()
selected_features.remove('target')

# Define features and target
X = df[selected_features]  # Selected Features
y = df['target']  # Target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize classifiers
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Naïve Bayes": GaussianNB(),
    "SVM": SVC(kernel='linear', random_state=42)
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")
    

# Save cleaned dataset
df.to_csv(r"C:\Users\Dell\Desktop\heart project\cleaned_data.csv", index=False)
print("Cleaned dataset saved as cleaned_data.csv")

# Display statistical summary
print("Statistical Summary:\n", df.describe())

# **1️⃣ Correlation Heatmap**
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# **2️⃣ Histograms & Boxplots**
features_to_visualize = ['age', 'trestbps', 'chol', 'thalach']
for feature in features_to_visualize:
    plt.figure(figsize=(12, 5))
    
    # Histogram
    plt.subplot(1, 2, 1)
    sns.histplot(df[feature], kde=True, bins=30)
    plt.title(f"Histogram of {feature}")
    
    # Boxplot
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[feature])
    plt.title(f"Boxplot of {feature}")
    
    plt.show()

# **3️⃣ Feature Importance Plot**
X = df.drop(columns=['target'])
y = df['target']

# Train a RandomForestClassifier to get feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

# Plot Feature Importance
plt.figure(figsize=(12, 6))
sns.barplot(x=importances, y=importances.index, palette="viridis")
plt.title("Feature Importance for Heart Disease Prediction")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

patient = {
    "trestbps": 85,  # Updated from BloodPressure
    "BMI": 32,
    "Age": 50,
   "chol": 200,  # Added Cholesterol attribute
    "exang": 1    # Added Smoking attribute
}

# Convert to DataFrame
patient_df = pd.DataFrame([patient])

# 📌 Step 3: Rule-Based Expert System (Experta)
class HeartDiseaseRisk(Fact):
    """Fact format for heart disease risk assessment"""
    age = Field(int, mandatory=True)
    bmi = Field(float, mandatory=True)
    chol = Field(int, mandatory=True)
    trestbps = Field(int, mandatory=True)
    exang = Field(int, mandatory=True)

class HeartDiseaseExpertSystem(KnowledgeEngine):

    @Rule(HeartDiseaseRisk(chol=P(lambda c: c > 240), age=P(lambda a: a > 50)))
    def high_risk_chol_age(self):
        print("🔴 High Cholesterol & Age > 50 → HIGH RISK")
        self.declare(Fact(risk="high"))

    @Rule(HeartDiseaseRisk(trestbps=P(lambda bp: bp > 140), exang=1))
    def high_risk_bp_smoking(self):
        print("🔴 High Blood Pressure & Smoker → HIGH RISK")
        self.declare(Fact(risk="high"))

    @Rule(HeartDiseaseRisk(bmi=P(lambda bmi: bmi < 25), exang=0))
    def low_risk_exercise(self):
        print("🟢 Regular Exercise & Healthy BMI → LOW RISK")
        self.declare(Fact(risk="low"))

    @Rule(HeartDiseaseRisk(age=P(lambda a: a > 60)))
    def high_risk_old_age(self):
        print("🔴 Age > 60 → HIGH RISK")
        self.declare(Fact(risk="high"))

    @Rule(HeartDiseaseRisk(bmi=P(lambda bmi: bmi > 30)))
    def high_risk_obesity(self):
        print("🔴 Obesity (BMI > 30) → HIGH RISK")
        self.declare(Fact(risk="high"))

    @Rule(Fact(risk="high"))
    def conclude_high_risk(self):
        print("⚠️ Final Assessment: HIGH RISK of Heart Disease")

    @Rule(Fact(risk="low"))
    def conclude_low_risk(self):
        print("✅ Final Assessment: LOW RISK of Heart Disease")

def get_user_input():
    return {
        "age": int(input("Enter Age: ")),
        "bmi": float(input("Enter BMI: ")),
        "chol": int(input("Enter Cholesterol Level: ")),
        "trestbps": int(input("Enter Blood Pressure (trestbps): ")),
        "exang": int(input("Enter 1 if Smoker, 0 if Non-Smoker: "))
    }

# Run Expert System
engine = HeartDiseaseExpertSystem()
engine.reset()
patient_data = get_user_input()
engine.declare(HeartDiseaseRisk(**patient_data))
engine.run()

# 📌 Step 4: Machine Learning Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Naïve Bayes": GaussianNB(),
    "SVM": SVC(kernel='linear', random_state=42)
    
}
# Split dataset (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Decision Tree model
dt = DecisionTreeClassifier(random_state=42)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters and model
best_dt = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Evaluate the optimized model
y_pred = best_dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

# Save the trained model
joblib.dump(best_dt, r"C:\Users\Dell\Desktop\heart project\decision_tree_model.pkl")
#print("Trained Decision Tree model saved as 'decision_tree_model.pkl'")

# Load trained Decision Tree model
dt_model = joblib.load(r"C:\Users\Dell\Desktop\heart project\decision_tree_model.pkl")


# Load Validation Data (Assume X_val and y_val are preprocessed)
X_val = pd.read_csv(r"C:\Users\Dell\Desktop\heart project\X_validation.csv")  # Validation features
y_val = pd.read_csv(r"C:\Users\Dell\Desktop\heart project\y_validation.csv")  # True labels

# Load Decision Tree Model
dt_model = joblib.load("decision_tree_model.pkl")  # Load trained model

expected_features = dt_model.feature_names_in_  # Get expected feature names from model
X_val = X_val.reindex(columns=expected_features, fill_value=0)  # Reorder and fill missing columns

# Predict using Decision Tree
try:
    y_pred_dt = dt_model.predict(X_val)
    print("Prediction Successful!")  # Debugging confirmation
except Exception as e:
    print(f"Error during prediction: {e}")

# Ensure y_val is correctly formatted
if isinstance(y_val, pd.DataFrame):
    y_val = y_val.squeeze()  # Convert to Series if it's a DataFrame

# Calculate Accuracy of Decision Tree
accuracy_dt = accuracy_score(y_val, y_pred_dt)
print(f"Decision Tree Accuracy: {accuracy_dt:.2f}")

# Print first few predictions for verification
print("Sample Predictions:", y_pred_dt[:10])

# Predict using Decision Tree
y_pred_dt = dt_model.predict(X_val)

# Calculate Accuracy of Decision Tree
accuracy_dt = accuracy_score(y_val, y_pred_dt)
print(f"Decision Tree Accuracy: {accuracy_dt:.2f}")

# Expert System Definition
class HeartDiseaseRisk(Fact):
    age = Field(int, mandatory=True)
    chol = Field(int, mandatory=True)
    trestbps = Field(int, mandatory=True)
    exang = Field(int, mandatory=True)

class HeartDiseaseExpert(KnowledgeEngine):
    def _init_(self):
        super()._init_()
        self.predictions = []  # Store results

    @Rule(HeartDiseaseRisk(age=P(lambda x: x > 50), chol=P(lambda x: x > 240)))
    def high_risk(self):
        self.predictions.append(1)  # High risk

    @Rule(HeartDiseaseRisk(age=P(lambda x: x <= 50), chol=P(lambda x: x <= 240)))
    def low_risk(self):
        self.predictions.append(0)  # Low risk

# Evaluate Expert System
engine = HeartDiseaseExpert()
y_pred_expert = []

for _, row in X_val.iterrows():
    row_dict = row.to_dict()

    # Convert required fields to int
    row_dict['age'] = int(row_dict['age'])
    row_dict['chol'] = int(row_dict['chol'])
    row_dict['trestbps'] = int(row_dict['trestbps'])
    row_dict['exang'] = int(row_dict['exang'])

    engine.reset()
    engine.predictions = []
    engine.declare(HeartDiseaseRisk(**row_dict))
    engine.run()

    # Get prediction (default to 0 if no rules matched)
    prediction = engine.predictions[0] if engine.predictions else 0
    y_pred_expert.append(prediction)

# Convert to numpy arrays
y_pred_expert = np.array(y_pred_expert)

# Calculate Accuracy of Expert System
accuracy_expert = accuracy_score(y_val, y_pred_expert)
print(f"Expert System Accuracy: {accuracy_expert:.2f}")

# Print Classification Report
print("\nDecision Tree Classification Report:")
print(classification_report(y_val, y_pred_dt))

print("\nExpert System Classification Report:")
print(classification_report(y_val, y_pred_expert))

# Compare Explainability
print("\nDecision Tree Rules:")
from sklearn.tree import export_text
tree_rules = export_text(dt_model, feature_names=list(X_val.columns))
print(tree_rules)

print("\nExpert System Rules:")
print("1. If age > 50 and cholesterol > 240 → High Risk")
print("2. If age ≤ 50 and cholesterol ≤ 240 → Low Risk")

# Conclusion
if accuracy_dt > accuracy_expert:
    print("\n📌 Decision Tree performed better!")
else:
    print("\n📌 Expert System performed better!")


for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")

# 📌 Step 5: Compare Expert System & Machine Learning
#print("\n🔍 **Comparison of Models & Expert System**")
#print("✅ Expert System provides rules-based risk assessment.")
#print("✅ Machine Learning provides prediction-based risk assessment.")
#print("⚖️ Use both for better results!")
# Predict using Random Forest (change model as needed)
prediction = models["Random Forest"].predict(patient_df)[0]
print("Prediction:", "Positive" if prediction == 1 else "Negative")

prediction = models["Naïve Bayes"].predict(patient_df)[0]
print("Prediction:", "Positive" if prediction == 1 else "Negative")

prediction = models["SVM"].predict(patient_df)[0]
print("Prediction:", "Positive" if prediction == 1 else "Negative")



# Define fuzzy variables
bmi = ctrl.Antecedent(np.arange(10, 50, 1), 'bmi')
age = ctrl.Antecedent(np.arange(10, 100, 1), 'age')
chol = ctrl.Antecedent(np.arange(100, 400, 1), 'chol')  # Added Cholesterol
exang = ctrl.Antecedent(np.arange(0, 2, 1), 'exang')  # Added Smoking

bmi['low'] = fuzz.trimf(bmi.universe, [10, 18, 25])
bmi['medium'] = fuzz.trimf(bmi.universe, [20, 30, 40])
bmi['high'] = fuzz.trimf(bmi.universe, [35, 45, 50])

age['young'] = fuzz.trimf(age.universe, [10, 20, 35])
age['middle'] = fuzz.trimf(age.universe, [30, 50, 70])
age['old'] = fuzz.trimf(age.universe, [60, 80, 100])

chol['low'] = fuzz.trimf(chol.universe, [100, 150, 200])
chol['medium'] = fuzz.trimf(chol.universe, [180, 220, 260])
chol['high'] = fuzz.trimf(chol.universe, [240, 300, 400])

exang['no'] = fuzz.trimf(exang.universe, [0, 0, 1])
exang['yes'] = fuzz.trimf(exang.universe, [1, 1, 1])

risk = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'risk')
risk['low'] = fuzz.trimf(risk.universe, [0, 0.2, 0.4])
risk['medium'] = fuzz.trimf(risk.universe, [0.3, 0.5, 0.7])
risk['high'] = fuzz.trimf(risk.universe, [0.6, 0.8, 1])

rule1 = ctrl.Rule(bmi['high'] & age['old'] & chol['high'] & exang['yes'], risk['high'])
rule2 = ctrl.Rule(bmi['medium'] & age['middle'] & chol['medium'] & exang['no'], risk['medium'])
rule3 = ctrl.Rule(bmi['low'] & age['young'] & chol['low'] & exang['no'], risk['low'])

# Control system
risk_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
risk_sim = ctrl.ControlSystemSimulation(risk_ctrl)

# Example patient data
risk_sim.input['bmi'] = 32
risk_sim.input['age'] = 50
risk_sim.input['chol'] = 200
risk_sim.input['exang'] = 1

risk_sim.compute()
print(f"Risk Score: {risk_sim.output['risk']:.2f}")