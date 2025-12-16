import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import sys

def load_and_debug():
    file_path = 'db_elasticidade.csv'
    with open('debug_clean.txt', 'w', encoding='utf-8') as f:
        f.write(f"Loading {file_path}...\n")
        try:
            # Read raw
            df_raw = pd.read_csv(file_path)
            f.write(f"Raw tail:\n{df_raw['kpi'].tail(15)}\n")
            
            # Apply logic
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
            
            f.write("\nCleaned Tail:\n")
            f.write(str(df[['quantity', 'price']].tail(15)) + "\n")
            
            df = df[(df['quantity'] > 0) & (df['price'] > 0)]
            df['ln_quantity'] = np.log(df['quantity'])
            df['ln_price'] = np.log(df['price'])
            
            model = smf.ols("ln_quantity ~ ln_price", data=df).fit()
            f.write(f"\nELASTICITY (Beta): {model.params['ln_price']}\n")
            
        except Exception as e:
            f.write(f"Error: {e}\n")

if __name__ == "__main__":
    load_and_debug()
