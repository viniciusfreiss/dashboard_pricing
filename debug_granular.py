import pandas as pd
import numpy as np
import statsmodels.api as sm

def load_granular_data():
    file_path = 'db_per_course.csv'
    try:
        df_g = pd.read_csv(file_path)
        print("File loaded successfully.")
    except FileNotFoundError:
        print("File not found.")
        return None
        
    print("Raw 'Valor Líquido' head:")
    print(df_g['Valor Líquido'].head())

    # Clean Currency logic test
    try:
        # Replicating the logic from dashboard.py
        df_g['price'] = df_g['Valor Líquido'].astype(str).str.replace('R$', '', regex=False).str.replace(',', '', regex=False)
        df_g['price'] = pd.to_numeric(df_g['price'], errors='coerce')
        print("\nCleaned 'price' head:")
        print(df_g['price'].head())
        print("\nPrice NaN count:", df_g['price'].isna().sum())
    except Exception as e:
        print(f"Error in cleaning: {e}")

    # Date
    try:
        df_g['date'] = pd.to_datetime(df_g['Data'], dayfirst=True)
        print("\nDate head:")
        print(df_g['date'].head())
    except Exception as e:
        print(f"Error in date parsing: {e}")
    
    # Rename
    df_g = df_g.rename(columns={'Tipo de Curso': 'type', 'UF': 'state', 'Forma de Pagamento': 'payment', 'Curso': 'course'})
    
    return df_g

df = load_granular_data()

if df is not None:
    print("\nAttempting Aggregation and Regression...")
    segment_by = 'type'
    unique_segments = df[segment_by].dropna().unique()
    print(f"Segments found: {unique_segments}")
    
    for seg in unique_segments:
        sub = df[df[segment_by] == seg]
        print(f"\nProcessing Segment: {seg}, Rows: {sub.shape[0]}")
        
        weekly = sub.set_index('date').resample('W').agg({'price': 'mean', 'type': 'count'}).rename(columns={'type': 'quantity'})
        weekly = weekly.dropna()
        weekly = weekly[weekly['quantity'] > 0]
        
        print(f"Weekly aggregated points: {len(weekly)}")
        if len(weekly) > 0:
            print(weekly.head())
