import pandas as pd

# Load data from the original dataset
file_path = 'Online_Retail.xlsx'  # Update this with the correct file path
data = pd.read_excel(file_path)

# Step 1: Drop rows with missing CustomerID
data = data.dropna(subset=['CustomerID'])

# Step 2: Fill missing Description with 'No Description'
data['Description'] = data['Description'].fillna('No Description')

# Step 3: Remove rows with non-positive Quantity values
data = data[data['Quantity'] > 0]

# Step 4: Convert InvoiceDate to date format (YYYY-MM-DD)
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate']).dt.strftime('%Y-%m-%d')

# Step 5: Apply outlier filtering
quantity_thresholds = data['Quantity'].quantile([0.01, 0.99])
unitprice_thresholds = data['UnitPrice'].quantile([0.01, 0.99])

data = data[
    (data['Quantity'].between(quantity_thresholds.iloc[0], quantity_thresholds.iloc[1])) &
    (data['UnitPrice'].between(unitprice_thresholds.iloc[0], unitprice_thresholds.iloc[1]))
]

# Step 6: Add TotalPrice column
data['TotalPrice'] = data['Quantity'] * data['UnitPrice']

# Save the cleaned data to a CSV file
output_file = 'Cleaned_Online_Retail_Corrected.csv'  # Update with desired output path
data.to_csv(output_file, index=False)

print(f"Cleaned data saved to: {output_file}")
