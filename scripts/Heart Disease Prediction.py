#Load the Dataset
import pandas as pd

# Use relative path
file_path = r"C:\Users\inder\OneDrive/Desktop/heart-disease-prediction/data/processed.cleveland.data"

# Load dataset
df = pd.read_csv(file_path, header=None)

# Save as CSV for easier access
df.to_csv(r"C:\Users\inder\OneDrive/Desktop/heart-disease-prediction/data/heart_disease.csv", index=False)

print("Dataset saved as heart_disease.csv!")
df.head()

#Assigning Column Names
df.columns = [
    "age", "sex", "chest_pain_type", "resting_bp", "cholesterol",
    "fasting_blood_sugar", "rest_ecg", "max_heart_rate", "exercise_angina",
    "st_depression", "st_slope", "num_major_vessels", "thalassemia", "target"
]

print(df.head()) 

df.shape
df.dtypes

#Replacing Missing or Invalid Values
import numpy as np  # Import numpy
# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# Convert columns to numeric, forcing errors to NaN
df = df.apply(pd.to_numeric, errors='coerce')
df.isnull().sum()
df.fillna(df.mean(), inplace=True)df['age'] = df['age'].astype(int)  
df['sex'] = df['sex'].astype(int)
df['chest_pain_type'] = df['chest_pain_type'].astype(int) 
df['fasting_blood_sugar'] = df['fasting_blood_sugar'].astype(int) 
df['rest_ecg'] = df['rest_ecg'].astype(int) 
df['exercise_angina'] = df['exercise_angina'].astype(int)  
df['st_slope'] = df['st_slope'].astype(int)
df['num_major_vessels'] = df['num_major_vessels'].astype(int)
df['thalassemia'] = df['thalassemia'].astype(int)

# again to check data types
df.dtypes
#Summary Statistics
df.describe()

#Exploratory Data Analysis (EDA)


import matplotlib.pyplot as plt
import seaborn as sns

#Age Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["age"], bins=20, kde=True)
plt.title("Age Distribution of Patients")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()


#Heart Disease Count by Sex
plt.figure(figsize=(6, 4))
sns.countplot(x="sex", hue="target", data=df, palette="coolwarm")
plt.title("Heart Disease Count by Sex")
plt.xticks(ticks=[0, 1], labels=["Female", "Male"])
plt.xlabel("Sex")
plt.ylabel("Count")
plt.show()

#Chest Pain Type vs. Heart Disease
plt.figure(figsize=(6, 4))
sns.countplot(x="chest_pain_type", hue="target", data=df, palette="viridis")
plt.title("Chest Pain Type vs. Heart Disease")
plt.xlabel("Chest Pain Type")
plt.ylabel("Count")
plt.show()

#Distribution of Target Variable
plt.figure(figsize=(6, 4))
sns.countplot(x="target", data=df, hue="target", palette="viridis", legend=False)
plt.xlabel("Heart Disease Presence")
plt.ylabel("Count")
plt.title("Distribution of Heart Disease Cases")
plt.show()

#Check for Data Imbalance
print(df["target"].value_counts())
df.groupby("target").mean()

# Select numerical columns only
numeric_cols = df.select_dtypes(include=["number"]).columns

#Feature Distributions
# Plot histograms
df[numeric_cols].hist(figsize=(12, 10), bins=20)
plt.show()
plt.figure(figsize=(10, 6))


#Correlation Matrix
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

#Applying SMOTE to Balance Classes
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split


X = df[['num_major_vessels', 'st_depression', 'st_slope', 'chest_pain_type',
            'exercise_angina', 'max_heart_rate', 'thalassemia', 'age', 'sex']] 
y = df['target']  


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance the classes in the training set
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)import xgboost as xgb


#Training XGBoost Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Scale features 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train XGBoost model
model = xgb.XGBClassifier(
    max_depth=10,               # Maximum depth of the tree
    n_estimators=100,           # Number of boosting rounds (trees)
    objective='multi:softmax',  # For multi-class classification
    num_class=9,                # Number of classes 
    eval_metric='mlogloss'      # Evaluation metric for multi-class classification
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Print detailed classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
from sklearn.model_selection import GridSearchCV

# Hyperparameters to tune
param_grid = {
    'max_depth': [5, 10, 15],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9],
    'min_child_weight': [1, 3, 5]
}

grid_search = GridSearchCV(estimator=xgb.XGBClassifier(objective='multi:softmax', num_class=5),
                           param_grid=param_grid, cv=3, verbose=2, n_jobs=-1, scoring='accuracy')

grid_search.fit(X_train, y_train)

print("Best Parameters: ", grid_search.best_params_)
print("Best Accuracy: ", grid_search.best_score_)
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score

# Predict on test data
y_pred = model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Classification Report
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)

#If applicable: ROC AUC score (for binary/multi-class classification)
y_prob = model.predict_proba(X_test)  # for multi-class, returns probabilities
auc = roc_auc_score(y_test, y_prob, multi_class='ovr', average='macro')  # 'macro' for averaging
print("AUC:", auc)