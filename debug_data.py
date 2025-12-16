import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

def load_and_debug():
    file_path = 'db_elasticidade.csv'
    print(f"Loading {file_path}...")
    try:
        # Read raw first to see what we are dealing with
        df_raw = pd.read_csv(file_path)
        print("Raw 'kpi' sample (tail):")
        print(df_raw['kpi'].tail(20))
        
        # Apply the logic I implemented
        def clean_val(val):
            if pd.isna(val): return 0.0
            if isinstance(val, (int, float)): return float(val)
            s = str(val).strip().replace('"', '').replace("'", "")
            
            # Case: "1,245,0"
            if s.count(',') > 1:
                parts = s.split(',')
                integer_part = "".join(parts[:-1])
                decimal_part = parts[-1]
                res = float(f"{integer_part}.{decimal_part}")
                print(f"DEBUG: Converted '{val}' -> {res}")
                return res
            
            if ',' in s and '.' in s: 
                s = s.replace(',', '') 
            elif ',' in s:
                s = s.replace(',', '.')
                
            try:
                return float(s)
            except:
                return 0.0

        df = df_raw.copy()
        df['quantity'] = df['kpi'].apply(clean_val)
        df['price'] = df['revenue_per_kpi'].apply(clean_val)
        
        # Check descriptive stats
        print("\nDescriptive Stats:")
        print(df[['quantity', 'price']].describe())
        
        df = df[(df['quantity'] > 0) & (df['price'] > 0)]
        df['ln_quantity'] = np.log(df['quantity'])
        df['ln_price'] = np.log(df['price'])
        
        model = smf.ols("ln_quantity ~ ln_price", data=df).fit()
        print("\nRegression Result:")
        print(model.summary())
        print(f"\nELASTICITY (Beta): {model.params['ln_price']}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    load_and_debug()
