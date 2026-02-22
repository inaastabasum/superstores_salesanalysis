import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("train.csv")

print(df.head())
print(df.isnull().sum())
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
df['Postal Code'] = df['Postal Code'].fillna('Unknown')
top_products = df.groupby('Product Name')['Sales'].sum().sort_values(ascending=False).head(5)

print("Total Sales:",df['Sales'].sum())
print("Top 5 Products:\n",df.groupby('Product Name')['Sales'].sum().sort_values(ascending=False).head(5))
print("Sales by Region:\n",df.groupby('Region')['Sales'].sum())
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)
print("Monthly Sales Trend:\n",df.groupby(df['Order Date'].dt.to_period('M'))['Sales'].sum())
duplicates = df.duplicated().sum()
print("\nNumber of duplicate rows:", duplicates)
if duplicates > 0:
    df = df.drop_duplicates()
    print("Duplicates removed. New dataset length:", len(df))
    df['Postal Code'] = df['Postal Code'].fillna('Unknown')
print("Missing Postal Codes handled.")
plt.figure(figsize=(10,6))  # Sets chart size
top_products.plot(kind='bar', color='skyblue')  # Bar chart
plt.title("Top 5 Products by Sales")  # Chart title
plt.xlabel("Product Name")  # X-axis label
plt.ylabel("Sales")  # Y-axis label
plt.xticks(rotation=45, ha='right')  # Rotate product names for readability
plt.tight_layout()  # Adjust layout so labels fit
plt.show() 
sales_region = df.groupby('Region')['Sales'].sum()
plt.figure(figsize=(8,5))
sales_region.plot(kind='bar', color='lightgreen')
plt.title("Sales by Region")
plt.xlabel("Region")
plt.ylabel("Sales")
plt.xticks(rotation=0)
plt.show()
monthly_sales = df.groupby(df['Order Date'].dt.to_period('M'))['Sales'].sum()
plt.figure(figsize=(12,6))
monthly_sales.plot(kind='line', marker='o', color='orange')
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.grid(True)  # Adds grid lines
plt.tight_layout()
plt.show()
print(df.columns)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
monthly_sales = monthly_sales.reset_index()
monthly_sales['Month_Number'] = np.arange(len(monthly_sales))

X = monthly_sales[['Month_Number']]
y = monthly_sales['Sales']

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

print("\nLinear Regression R² score (Monthly Trend):", r2)