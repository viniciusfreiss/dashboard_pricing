import pandas as pd
import numpy as np

def load_latest_baseline(file_path):
    df = pd.read_csv(file_path)
    # Use the average of the last 4 weeks as baseline to smooth out noise
    last_4_weeks = df.tail(4)
    baseline_price = last_4_weeks['revenue_per_kpi'].mean()
    baseline_kpi = last_4_weeks['kpi_clean'].mean()
    return baseline_price, baseline_kpi

def simulate_scenarios(baseline_price, baseline_quantity, elasticity):
    scenarios = []
    # Test price changes from -20% to +20% in 5% increments
    changes = np.arange(-0.20, 0.25, 0.05)
    
    for change in changes:
        new_price = baseline_price * (1 + change)
        # Q_new = Q_old * (P_new / P_old) ^ elasticity
        new_quantity = baseline_quantity * ((new_price / baseline_price) ** elasticity)
        new_revenue = new_price * new_quantity
        
        baseline_revenue = baseline_price * baseline_quantity
        rev_change = (new_revenue - baseline_revenue) / baseline_revenue
        
        scenarios.append({
            'Price Change (%)': change * 100,
            'New Price': new_price,
            'Predicted Quantity': new_quantity,
            'Predicted Revenue': new_revenue,
            'Revenue Change (%)': rev_change * 100
        })
    
    return pd.DataFrame(scenarios)

def main():
    data_path = r'c:\Users\vfelisberto\Desktop\elasticidade\cleaned_data.csv'
    elasticity = -1.1478 # From analysis
    
    try:
        base_price, base_qty = load_latest_baseline(data_path)
    except FileNotFoundError:
        print("Cleaned data not found. Please run elasticity_analysis.py first.")
        return

    print(f"=== Planning Model Baseline (Last 4 Weeks Avg) ===")
    print(f"Baseline Price: {base_price:,.2f}")
    print(f"Baseline Quantity: {base_qty:,.1f}")
    print(f"Elasticity Used: {elasticity}")
    print("-" * 50)
    
    df_scenarios = simulate_scenarios(base_price, base_qty, elasticity)
    
    # Formatting for display
    pd.options.display.float_format = '{:,.2f}'.format
    print("\n=== Price Change Scenarios ===")
    print(df_scenarios)
    
    # Save to CSV
    output_path = r'c:\Users\vfelisberto\Desktop\elasticidade\planning_scenarios.csv'
    df_scenarios.to_csv(output_path, index=False)
    print(f"\nScenarios saved to {output_path}")
    
    # Recommendation
    print("\n=== Recommendation ===")
    if elasticity < -1:
        print("Demand is ELASTIC (|e| > 1). Decreasing price will likely INCREASE total revenue.")
    elif elasticity > -1:
        print("Demand is INELASTIC (|e| < 1). Increasing price will likely INCREASE total revenue.")
    else:
        print("Demand is UNIT ELASTIC (|e| = 1). Revenue is maximized at current price.")

if __name__ == "__main__":
    main()
