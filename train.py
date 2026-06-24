import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

def load_and_preprocess_data(filepath):
    print(f"[*] Loading dataset from: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
        
    df = pd.read_csv(filepath)
    
    print("[*] Preprocessing dataset...")
    # Clean TotalCharges column
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    median_val = df['TotalCharges'].median()
    df['TotalCharges'] = df['TotalCharges'].fillna(median_val)
    
    return df

def train_and_evaluate(df):
    print("[*] Splitting dataset into train/test (80/20)...")
    
    # Drop IDs and Target
    X = df.drop(['customerID', 'Churn'], axis=1, errors='ignore')
    # Map target
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define categorical and numerical features
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_features = [col for col in X.columns if col not in numeric_features]
    
    # Create preprocessing steps
    print("[*] Configuring preprocessing with OneHotEncoding and Selective Scaling...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
        ])
    
    # We will define two pipelines, one for RF, one for XGB
    pipelines = {
        'RandomForest': ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', RandomForestClassifier(random_state=42))
        ]),
        'XGBoost': ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', XGBClassifier(random_state=42, eval_metric='logloss'))
        ])
    }
    
    # Hyperparameters for RandomizedSearchCV
    param_grids = {
        'RandomForest': {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [None, 10, 20, 30],
            'classifier__min_samples_split': [2, 5, 10]
        },
        'XGBoost': {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__learning_rate': [0.01, 0.05, 0.1, 0.2],
            'classifier__max_depth': [3, 5, 7]
        }
    }
    
    best_models = {}
    
    for model_name in pipelines.keys():
        print(f"\n[*] Tuning and training {model_name}...")
        search = RandomizedSearchCV(
            pipelines[model_name], 
            param_distributions=param_grids[model_name], 
            n_iter=5, 
            cv=3, 
            scoring='roc_auc', 
            random_state=42, 
            n_jobs=-1
        )
        search.fit(X_train, y_train)
        best_models[model_name] = search.best_estimator_
        print(f"[+] Best params for {model_name}: {search.best_params_}")
    
    print("\n[*] Evaluating models...")
    os.makedirs('images', exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    
    best_roc_auc = 0
    best_model_name = ""
    best_y_pred = None
    
    for model_name, model in best_models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)
        
        print(f"\n--- {model_name} Performance ---")
        print(f"Accuracy: {acc:.4f}")
        print(f"ROC-AUC:  {roc:.4f}")
        print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))
        
        if roc > best_roc_auc:
            best_roc_auc = roc
            best_model_name = model_name
            best_y_pred = y_pred
            
        # Plot ROC
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc:.3f})')
        
    # Finalize ROC plot
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    roc_path = os.path.join('images', 'roc_curves.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved ROC curves plot to: {roc_path}")
    
    # Save Confusion Matrix for the best model
    print(f"\n[*] Generating confusion matrix for best model ({best_model_name})...")
    cm = confusion_matrix(y_test, best_y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    
    plt.figure(figsize=(6, 5))
    disp.plot(cmap="coolwarm", ax=plt.gca())
    plt.title(f'Confusion Matrix - {best_model_name}')
    plot_path = os.path.join('images', 'confusion_matrix.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved best model confusion matrix plot to: {plot_path}")
    
    # Plot Feature Importances for Best Model
    try:
        best_pipeline = best_models[best_model_name]
        classifier = best_pipeline.named_steps['classifier']
        
        # Get feature names from ColumnTransformer
        preprocessor = best_pipeline.named_steps['preprocessor']
        ohe_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        all_features = numeric_features + list(ohe_features)
        
        importances = classifier.feature_importances_
        indices = np.argsort(importances)[::-1][:15] # Top 15
        
        plt.figure(figsize=(10, 6))
        plt.title(f"Top 15 Feature Importances ({best_model_name})")
        plt.barh(range(15), importances[indices][:15][::-1], color="skyblue", align="center")
        plt.yticks(range(15), [all_features[i] for i in indices][:15][::-1])
        plt.xlabel("Relative Importance")
        feature_path = os.path.join('images', 'feature_importance.png')
        plt.savefig(feature_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] Saved feature importances plot to: {feature_path}")
    except Exception as e:
        print(f"[-] Could not plot feature importance: {e}")
        
    return best_model_name, best_roc_auc

if __name__ == '__main__':
    # Local path of dataset in repo folder
    data_path = os.path.join('data', 'Telco-Customer-Churn.csv')
    
    # Fallback to Downloads if not copied yet
    if not os.path.exists(data_path):
        data_path = r'C:\Users\JONAH A\Downloads\Telco-Customer-Churn.csv'
        
    try:
        data = load_and_preprocess_data(data_path)
        best_model_name, best_roc = train_and_evaluate(data)
        print(f"\n[+] Churn Prediction Pipeline executed successfully! Best Model: {best_model_name} (AUC: {best_roc:.4f})")
    except Exception as e:
        print(f"[-] Error: {e}")
