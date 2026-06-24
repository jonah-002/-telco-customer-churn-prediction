import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def load_and_preprocess_data(filepath):
    """
    Loads dataset and performs basic cleaning:
    - Coerces TotalCharges to numeric
    - Imputes missing values with median
    """
    print(f"[*] Loading dataset from: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
        
    df = pd.read_csv(filepath)
    
    # Clean TotalCharges column
    print("[*] Preprocessing TotalCharges...")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    median_val = df['TotalCharges'].median()
    df['TotalCharges'].fillna(median_val, inplace=True)
    
    return df

def encode_features(df):
    """
    Encodes categorical features using LabelEncoder
    """
    print("[*] Encoding categorical features...")
    df_encoded = df.copy()
    label_encoder = LabelEncoder()
    
    categorical_cols = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
        'PaperlessBilling', 'PaymentMethod', 'Churn'
    ]
    
    for col in categorical_cols:
        if col in df_encoded.columns:
            df_encoded[col] = label_encoder.fit_transform(df_encoded[col])
            
    return df_encoded

def train_and_evaluate(df):
    """
    Splits features/target, normalizes, trains Random Forest model, and evaluates accuracy
    """
    print("[*] Splitting dataset into train/test (80/20)...")
    # Separate features and target
    X = df.drop(['customerID', 'Churn'], axis=1, errors='ignore')
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Scale features
    print("[*] Scaling features using StandardScaler...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Initialize and train RandomForestClassifier
    print("[*] Training RandomForestClassifier model...")
    clf = RandomForestClassifier(random_state=0)
    clf.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test_scaled)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"[+] Model Accuracy: {accuracy:.4f}")
    
    # Save Confusion Matrix plot
    print("[*] Generating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
    
    plt.figure(figsize=(6, 5))
    disp.plot(cmap="coolwarm", ax=plt.gca())
    plt.title('Churn Prediction Confusion Matrix')
    
    # Ensure output folder for images exists
    os.makedirs('images', exist_ok=True)
    plot_path = os.path.join('images', 'confusion_matrix.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved confusion matrix plot to: {plot_path}")
    
    return accuracy

if __name__ == '__main__':
    # Local path of dataset in repo folder
    data_path = os.path.join('data', 'Telco-Customer-Churn.csv')
    
    # Fallback to Downloads if not copied yet
    if not os.path.exists(data_path):
        data_path = r'C:\Users\JONAH A\Downloads\Telco-Customer-Churn.csv'
        
    try:
        data = load_and_preprocess_data(data_path)
        encoded_data = encode_features(data)
        train_and_evaluate(encoded_data)
        print("[+] Churn Prediction Pipeline executed successfully!")
    except Exception as e:
        print(f"[-] Error: {e}")
