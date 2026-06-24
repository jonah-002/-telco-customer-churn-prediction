import os
import pandas as pd

def main():
    # Load dataset
    data_path = os.path.join('data', 'Telco-Customer-Churn.csv')
    if not os.path.exists(data_path):
        data_path = r'C:\Users\JONAH A\Downloads\Telco-Customer-Churn.csv'
        
    print(f"[*] Loading dataset from {data_path}")
    df = pd.read_csv(data_path)
    
    # Convert Churn to numeric for calculation (Yes -> 1, No -> 0)
    df['Churn_numeric'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    # Create a pivot table: Churn rate by Contract type and Internet Service
    print("[*] Generating Pivot Table: Churn Rate by Contract and Internet Service...")
    pivot = pd.pivot_table(
        df, 
        values='Churn_numeric', 
        index='Contract', 
        columns='InternetService', 
        aggfunc='mean'
    )
    
    pivot_pct = pivot.map(lambda x: f"{x*100:.1f}%" if pd.notnull(x) else "N/A")
    
    print("\n--- Churn Rate Pivot Table ---")
    print(pivot_pct)
    print("------------------------------\n")
    
    # Save the pivot table to a Markdown file
    out_path = 'churn_rate_pivot_table.md'
    with open(out_path, 'w') as f:
        f.write("# Churn Rate by Contract and Internet Service\n\n")
        f.write(pivot_pct.to_markdown())
    print(f"[+] Pivot table saved to {out_path}")

if __name__ == '__main__':
    main()
