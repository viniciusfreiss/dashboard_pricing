import pandas as pd

def analyze_payment():
    try:
        df = pd.read_csv('db_per_course.csv')
        print("--- Parcelas Unique Values ---")
        print(df['Parcelas'].unique())
        
        print("\n--- Forma de Pagamento Unique Values ---")
        print(df['Forma de Pagamento'].unique())
        
        print("\n--- Cross Tabulation (Parcelas vs Payment) ---")
        print(pd.crosstab(df['Parcelas'], df['Forma de Pagamento']))
        
        # Check 'À vista' equivalent
        print("\n--- Sample of 1x or 0x ---")
        print(df[df['Parcelas'].isin(['1x', '0x', '1', '0'])].head())

    except Exception as e:
        print(e)

if __name__ == "__main__":
    analyze_payment()
