# Telco Customer Churn Prediction

Predicting telecom customer churn using machine learning classifiers built on Scikit-Learn.

---

## 📋 Overview
Customer churn is one of the most critical metrics for telecom providers. Retaining existing customers is significantly more cost-effective than acquiring new ones. This repository implements an end-to-end machine learning pipeline to preprocess customer data, perform exploratory data analysis, and train a Random Forest model to predict churn.

The project currently achieves a **79% prediction accuracy** using a Random Forest Classifier on test data.

---

## 📁 Repository Structure
```bash
├── data/
│   └── Telco-Customer-Churn.csv   # Dataset (to be placed here)
├── images/
│   └── confusion_matrix.png       # Saved confusion matrix evaluation plot
├── churn_prediction_notebook.ipynb # Interactive Jupyter Notebook analysis
├── train.py                        # Clean, modular training script
├── requirements.txt                # Project dependencies
└── .gitignore                      # Git exclusion rules
```

---

## 🛠️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/jonah-002/telco-customer-churn-prediction.git
cd telco-customer-churn-prediction
```

### 2. Install Dependencies
Ensure you have Python 3.8+ installed. Install the required libraries via `pip`:
```bash
pip install -r requirements.txt
```

### 3. Place the Dataset
Download the dataset from [Kaggle (Telco Customer Churn)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place the CSV in the `data/` folder:
* **Destination path**: `data/Telco-Customer-Churn.csv`

---

## 🚀 How to Run

### Run the Modular Script
Execute the main training script to clean the data, train the Random Forest Classifier, and generate validation charts:
```bash
python train.py
```
This script will output accuracy scores directly to your console and output a confusion matrix plot under the `images/` directory.

### Run the Interactive Notebook
To interactively explore the data step-by-step:
```bash
jupyter notebook churn_prediction_notebook.ipynb
```

---

## 📊 Dataset Features
The dataset contains information about **7,043** customers across 21 columns. The primary target is **Churn** (whether the customer left within the last month).

Key feature groups:
* **Demographics**: `gender`, `SeniorCitizen`, `Partner`, `Dependents`.
* **Services Signed Up**: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`.
* **Account Info**: `tenure`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`.

---

## 📈 Preprocessing and Modeling Pipeline
1. **Numeric Coercion**: Convert `TotalCharges` to float, coercing empty spaces to `NaN`.
2. **Imputation**: Missing values in `TotalCharges` are filled with the column's median.
3. **Encoding**: Categorical string features are encoded into numerical values using `LabelEncoder`.
4. **Data Splitting**: Split into 80% training / 20% test partitions.
5. **Standard Scaling**: Continuous feature columns are standard-scaled to ensure equal variance.
6. **Training**: A Scikit-Learn `RandomForestClassifier` is trained on training data.
7. **Evaluation**: Predictions are checked against test targets using Accuracy and Confusion Matrix metrics.

---

## 📌 Model Evaluation Results
* **Accuracy Score**: `79.0%`
* **Confusion Matrix**:

![Confusion Matrix](images/confusion_matrix.png)

---

## 💡 Planned Enhancements
* **One-Hot Encoding**: Replace ordinal LabelEncoding on nominal columns to improve generalization.
* **Class Imbalance Management**: Utilize SMOTE oversampling or incorporate class weights (`class_weight="balanced"`) to boost true-positive rates.
* **Hyperparameter Tuning**: Run RandomizedSearchCV to optimize Random Forest estimators.
