import pandas as pd

def check_cash_prices():
    try:
        df = pd.read_csv('db_per_course.csv')
        
        # Robust Cleaning (Same as Dashboard)
        def clean_val(val):
            if pd.isna(val): return 0.0
            if isinstance(val, (int, float)): return float(val)
            s = str(val).strip().replace('"', '').replace("'", "").replace("R$", "").strip()
            if s.count(',') > 1:
                parts = s.split(',')
                integer_part = "".join(parts[:-1])
                decimal_part = parts[-1]
                return float(f"{integer_part}.{decimal_part}")
            if ',' in s and '.' in s: 
                s = s.replace(',', '') 
            elif ',' in s:
                s = s.replace(',', '.')
            try:
                return float(s)
            except:
                return 0.0

        df['transaction_value'] = df['Valor Líquido'].apply(clean_val)
        
        # Filter for BIM
        df_bim = df[df['Curso'].astype(str).str.upper().str.contains("BUILDING INFORMATION MODELING")].copy()
        
        print(f"BIM Total Rows: {len(df_bim)}")
        
        # Filter for Cash
        df_cash = df_bim[df_bim['Parcelas'].str.contains("vista", case=False, na=False)]
        
        print(f"BIM Cash Rows: {len(df_cash)}")
        if not df_cash.empty:
            print("Cash Price Stats:")
            print(df_cash['transaction_value'].describe())
            print("\nTop 5 Lowest Cash Prices:")
            print(df_cash.sort_values('transaction_value').head(5)[['Data', 'Valor Líquido', 'transaction_value']])
        else:
            print("No Cash transactions found.")

    except Exception as e:
        print(e)

if __name__ == "__main__":
    check_cash_prices()
