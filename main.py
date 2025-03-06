import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib  # For saving and loading the model

# 1. Load data
try:
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    stroke_df = pd.read_csv("healthcare-dataset-stroke-data.csv")
except FileNotFoundError as e:
    print(f"Error: File not found: {e}")
    exit()
except Exception as e:
    print(f"Error reading file: {e}")
    exit()

# 2. Data preprocessing

## 2.1 Concatenate data for unified processing
all_data = pd.concat([train_df, test_df, stroke_df], ignore_index=True)

## 2.2 Handle missing values
# Check for missing values
print("Missing value statistics:")
print(all_data.isnull().sum())

# Fill missing values in 'bmi' column with the mean
all_data['bmi'] = all_data['bmi'].fillna(all_data['bmi'].mean())

## 2.3 Encode categorical variables
# Use LabelEncoder to encode 'gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'
label_encoder = LabelEncoder()
all_data['gender'] = label_encoder.fit_transform(all_data['gender'])
all_data['ever_married'] = label_encoder.fit_transform(all_data['ever_married'])
all_data['work_type'] = label_encoder.fit_transform(all_data['work_type'])
all_data['Residence_type'] = label_encoder.fit_transform(all_data['Residence_type'])
all_data['smoking_status'] = label_encoder.fit_transform(all_data['smoking_status'])

## 2.4 Feature scaling
# Use StandardScaler to scale numerical features, which helps XGBoost performance
numerical_features = ['age', 'avg_glucose_level', 'bmi', 'heart_disease', 'hypertension']
scaler = StandardScaler()
all_data[numerical_features] = scaler.fit_transform(all_data[numerical_features])

# 3. Prepare training and testing data

## 3.1 Split data
# Assume 'id' column exists only in train_df and test_df to distinguish data sources
train_data = all_data[all_data['id'].isin(train_df['id'])].drop('id', axis=1)
test_data = all_data[all_data['id'].isin(test_df['id'])].drop('id', axis=1)
stroke_data = all_data[~all_data['id'].isin(pd.concat([train_df['id'], test_df['id']]))].drop('id', axis=1)

## 3.2 Split training and validation sets
X = stroke_data.drop('stroke', axis=1)
y = stroke_data['stroke']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train XGBoost model
# Initialize XGBoost classifier
xgb_classifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Train the model
xgb_classifier.fit(X_train, y_train)

# 5. Evaluate the model
# Make predictions on the validation set
y_pred = xgb_classifier.predict(X_val)
y_prob = xgb_classifier.predict_proba(X_val)[:, 1]  # Get probabilities for the positive class

# Calculate accuracy
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation set accuracy: {accuracy:.4f}")

# Calculate AUROC
auroc = roc_auc_score(y_val, y_prob)
print(f"Validation set AUROC: {auroc:.4f}")

# Print classification report
print("Classification report:")
print(classification_report(y_val, y_pred))

# 6. Make predictions on the test data
X_test = test_data.drop('stroke', axis=1, errors='ignore') # Ignore error if 'stroke' column is missing
# Ensure the test set has the same columns as the training set
missing_cols = set(X_train.columns) - set(X_test.columns)
for c in missing_cols:
    X_test[c] = 0
# Ensure the column order is the same
X_test = X_test[X_train.columns]

test_pred = xgb_classifier.predict(X_test)

# 7. Save the model
model_filename = "xgboost_stroke_model.joblib"
joblib.dump(xgb_classifier, model_filename)
print(f"Model saved as {model_filename}")

# 8. (Optional) Load the model
# loaded_model = joblib.load("xgboost_stroke_model.joblib")