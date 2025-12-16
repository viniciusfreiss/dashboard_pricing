import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

def clean_kpi(val):
    if isinstance(val, (int, float)):
        return float(val)
    val = str(val).strip()
    # Handle "1,327,0" format
    if ',' in val:
        # If it ends with ,0 it's likely a decimal zero
        if val.endswith(',0'):
            val = val[:-2] # Remove the ,0
        # Remove thousands separators
        val = val.replace(',', '')
    return float(val)

def main():
    file_path = r'c:\Users\vfelisberto\Desktop\elasticidade\db_elasticidade.csv'
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Clean KPI
    df['kpi_clean'] = df['kpi'].apply(clean_kpi)
    
    # Clean Revenue per KPI (just in case, though it looked fine)
    # It seemed to be standard floats in the preview, but good to be safe
    # df['revenue_per_kpi'] is likely already float
    
    # Drop rows with 0 or NaN to avoid log errors
    df = df[(df['kpi_clean'] > 0) & (df['revenue_per_kpi'] > 0)].copy()
    
    # Log-Log Transformation
    df['ln_kpi'] = np.log(df['kpi_clean'])
    df['ln_price'] = np.log(df['revenue_per_kpi'])
    
    # Regression: ln(Q) = alpha + beta * ln(P)
    X = df['ln_price']
    X = sm.add_constant(X)
    y = df['ln_kpi']
    
    model = sm.OLS(y, X).fit()
    
    elasticity = model.params['ln_price']
    r_squared = model.rsquared
    
    report = []
    report.append("=== Elasticity Analysis Results ===")
    report.append(f"Elasticity (Price Sensitivity): {elasticity:.4f}")
    report.append(f"R-squared: {r_squared:.4f}")
    report.append("\nModel Summary:")
    report.append(str(model.summary()))
    
    with open(r'c:\Users\vfelisberto\Desktop\elasticidade\analysis_report.txt', 'w') as f:
        f.write('\n'.join(report))
        
    print('\n'.join(report))
    
    # Save cleaned data for the planning model
    df.to_csv(r'c:\Users\vfelisberto\Desktop\elasticidade\cleaned_data.csv', index=False)
    
    # Simple Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['revenue_per_kpi'], df['kpi_clean'], alpha=0.6)
    plt.title('Demand Curve: Price vs Quantity')
    plt.xlabel('Price (Revenue per KPI)')
    plt.ylabel('Quantity (KPI)')
    plt.grid(True)
    plt.savefig(r'c:\Users\vfelisberto\Desktop\elasticidade\demand_curve.png')
    print("\nSaved demand curve plot to demand_curve.png")

if __name__ == "__main__":
    main()
