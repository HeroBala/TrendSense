import pandas as pd

# Step 1: Load the JSON data
df = pd.read_json("data/ecommerce_full.json")

# Step 2: Show the first few rows as table
print("\n🧾 First 5 rows of your data:\n")
print(df.head())

# Step 3: Optionally, save to Excel format
excel_path = "data/ecommerce_full.xlsx"
df.to_excel(excel_path, index=False)

print(f"\n✅ Data also saved to Excel file: {excel_path}")
print("📂 You can now open this file with Excel, Numbers, or Google Sheets.")

