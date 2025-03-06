rubing:# Stroke Prediction Using XGBoost

## Overview
This project focuses on **predicting stroke risk** using **machine learning** techniques on healthcare data. It employs **XGBoost**, a powerful gradient boosting algorithm, to analyze patient characteristics and identify individuals at risk of stroke.

The dataset used includes **demographic, lifestyle, and clinical** factors, ensuring a comprehensive approach to stroke prediction. The model is trained on real-world healthcare data, ensuring accurate and actionable insights.

---

## 📌 Implementation Details
The project includes:
- **Data Preprocessing:** Handling missing values, encoding categorical variables, and scaling numerical features.
- **Model Training:** Using **XGBoost** for stroke prediction.
- **Model Evaluation:** Accuracy, AUROC score, and classification report.
- **Model Deployment:** Saving the trained model for future use.

### **Files in this repository:**
- `main.py`: Main script for preprocessing, training, and evaluating the model.
- `healthcare-dataset-stroke-data.csv`: Dataset containing patient health records.
- `train.csv`: Training dataset.
- `test.csv`: Test dataset.
- `xgboost_stroke_model.joblib`: Saved trained model.

---

## 🚀 Model Execution Workflow
### **1️⃣ Load and Preprocess Data**
The script loads **three datasets** (train, test, and stroke data) and performs preprocessing:
- **Missing values** are handled (e.g., filling BMI missing values with the mean).
- **Categorical variables** are encoded using **LabelEncoder**.
- **Numerical features** are standardized using **StandardScaler**.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load data
stroke_df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Handle missing values
stroke_df['bmi'] = stroke_df['bmi'].fillna(stroke_df['bmi'].mean())

# Encode categorical features
label_encoder = LabelEncoder()
stroke_df['gender'] = label_encoder.fit_transform(stroke_df['gender'])
stroke_df['ever_married'] = label_encoder.fit_transform(stroke_df['ever_married'])
```

### **2️⃣ Train-Test Split**
The dataset is split into **training** and **validation** sets:
```python
X = stroke_df.drop('stroke', axis=1)
y = stroke_df['stroke']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
```

### **3️⃣ Train XGBoost Model**
XGBoost is trained to classify stroke risk:
```python
from xgboost import XGBClassifier

xgb_classifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_classifier.fit(X_train, y_train)
```

### **4️⃣ Evaluate Model Performance**
Model performance is assessed using **accuracy** and **AUROC** scores:
```python
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

y_pred = xgb_classifier.predict(X_val)
y_prob = xgb_classifier.predict_proba(X_val)[:, 1]

accuracy = accuracy_score(y_val, y_pred)
auroc = roc_auc_score(y_val, y_prob)

print(f"Accuracy: {accuracy:.4f}")
print(f"AUROC Score: {auroc:.4f}")
print(classification_report(y_val, y_pred))
```

### **5️⃣ Save and Load Model**
```python
import joblib

# Save model
joblib.dump(xgb_classifier, "xgboost_stroke_model.joblib")

# Load model
loaded_model = joblib.load("xgboost_stroke_model.joblib")
```

---

## 🔄 Running the Model
### **1️⃣ Install Dependencies**
```bash
pip install pandas numpy scikit-learn xgboost joblib
```

### **2️⃣ Run the Script**
```bash
python main.py
```

### **3️⃣ Review Model Predictions**
Predictions on the test set are stored for further analysis.

---

## 📜 Future Enhancements
- Improve feature engineering with **additional medical indicators**.
- Fine-tune **hyperparameters** for better model performance.
- Deploy the model using **Flask or FastAPI** for real-time predictions.

---

## 📩 Contact
For further inquiries, reach out via the project discussion forum.

Building better stroke prediction models with AI! 🚀

