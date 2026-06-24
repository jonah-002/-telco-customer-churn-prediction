# Project Handoff: Telco Customer Churn Prediction

## 1. Project Overview
This project contains an end-to-end machine learning pipeline for predicting customer churn using the Telco Customer Churn dataset. It features data preprocessing, EDA, feature scaling, model training using a Random Forest Classifier, and an analytical pivot table script.

**Repository URL**: [https://github.com/jonah-002/-telco-customer-churn-prediction](https://github.com/jonah-002/-telco-customer-churn-prediction)

## 2. Current State and Achievements
* **Machine Learning Pipeline (`train.py`)**: A fully modularized script that reads the dataset, cleans it (handling missing `TotalCharges`), encodes categorical variables via `LabelEncoder`, scales numerical features, and trains a `RandomForestClassifier`.
* **Accuracy**: The model currently achieves an accuracy of approximately **79%**.
* **Visualizations**: A confusion matrix is automatically generated and saved to `images/confusion_matrix.png` during training.
* **Jupyter Notebook (`churn_prediction_notebook.ipynb`)**: An interactive version of the analysis, kept for reference and exploratory data analysis.
* **Pivot Table Analysis (`generate_pivot.py`)**: A supplementary script that computes churn rates grouped by `Contract` and `InternetService` and exports the result to `churn_rate_pivot_table.md`.
* **Project Infrastructure**: The repository is fully initialized with a descriptive `README.md`, `requirements.txt` for easy environment setup, and a `.gitignore` to prevent raw data and cached files from being committed.

## 3. Directory Structure
```
telco-customer-churn-prediction/
├── data/                            # Raw dataset goes here (ignored in git)
├── images/                          # Output directory for plots (e.g., confusion matrix)
├── .gitignore                       # Git exclusion rules
├── churn_prediction_notebook.ipynb  # Interactive data analysis notebook
├── churn_rate_pivot_table.md        # Exported pivot table of churn rates
├── generate_pivot.py                # Script to generate the pivot table
├── HANDOFF.md                       # This handoff document
├── README.md                        # Project documentation
├── requirements.txt                 # Python dependencies
└── train.py                         # Main ML training pipeline script
```

## 4. How to Run
1. Install dependencies: `pip install -r requirements.txt`
2. Ensure the dataset `Telco-Customer-Churn.csv` is placed inside the `data/` directory.
3. Run the main pipeline: `python train.py`
4. Run the pivot table analysis: `python generate_pivot.py`

## 5. Next Steps & Recommendations for Future Developers
* **Feature Engineering**: Improve categorical encoding. Currently, nominal variables like `PaymentMethod` or `InternetService` are encoded with `LabelEncoder`, which implies an ordinal relationship. Switching to One-Hot Encoding (`pd.get_dummies` or `OneHotEncoder`) should improve model logic.
* **Data Imbalance**: The dataset has a significant class imbalance (~73% No Churn vs. 27% Churn). Implement SMOTE or use `class_weight='balanced'` in the Random Forest to improve the recall of the minority class.
* **Selective Scaling**: Standard scaling is currently applied to all features. Modify the pipeline to scale only continuous numerical columns (`tenure`, `MonthlyCharges`, `TotalCharges`) while leaving binary flags intact.
* **Hyperparameter Tuning**: Use `GridSearchCV` or `RandomizedSearchCV` to optimize the Random Forest parameters (e.g., `n_estimators`, `max_depth`).
* **Alternative Models**: Experiment with gradient boosting models (XGBoost, LightGBM, CatBoost) or Logistic Regression for performance comparison.

## 6. Key Contacts
* **Project Owner**: Jonah Alexander (jonah.alexander02@gmail.com)
* **GitHub**: [@jonah-002](https://github.com/jonah-002)
