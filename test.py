import pandas as pd
import os

list_dir = os.listdir()
csv_files = [f for f in list_dir if f.endswith('.csv')]

for file in csv_files:
    try:
        df = pd.read_csv(file, nrows=3)
        columns = "-".join(df.columns)
        print(f"\ncsv name: {file}")
        print(f"columns: {columns}")
        print("sample data:")
        print(df.to_string(index=False))
        print("-" * 50)
    except Exception as e:
        print(f"{file}: خطا در خواندن ({e})\n")
