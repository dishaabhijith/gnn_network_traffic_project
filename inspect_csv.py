import pandas as pd

df = pd.read_csv('data/Midterm_53_group.csv')  # Adjust path if needed

print("Columns in CSV:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())
