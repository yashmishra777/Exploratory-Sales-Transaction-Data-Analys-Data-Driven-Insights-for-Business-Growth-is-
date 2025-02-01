# Step 1: Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Step 2: Load the Data
data_path = "sales_transaction_dataset.csv"
df = pd.read_csv('sales_transaction_dataset.csv')

# Step 3: Data Overview
print("Dataset Information:\n")
print(df.info())

print("\nSummary Statistics:\n")
print(df.describe())

print("\nMissing Values:\n")
print(df.isnull().sum())

print("\nUnique Values per Column:\n")
print(df.nunique())

print("\nDuplicate Rows:", df.duplicated().sum())

# Step 4: Handle Missing and Duplicate Data
# Drop duplicates
df = df.drop_duplicates()

# Fill missing values
for col in df.select_dtypes(include=['object']):
    df[col].fillna(df[col].mode()[0], inplace=True)

for col in df.select_dtypes(include=['number']):
    df[col].fillna(df[col].mean(), inplace=True)

# Step 5: Feature Engineering
# Convert date columns to datetime format
date_columns = ['Order_date', 'Ship_date']
for col in date_columns:
    df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')

# Drop rows with invalid dates
df = df.dropna(subset=date_columns)

# Create new features
df['Actual_shipping_delay'] = (df['Ship_date'] - df['Order_date']).dt.days
df['Profit_margin'] = df['Profit_per_order'] / df['Sales_per_order']
df['Order_year'] = df['Order_date'].dt.year
df['Order_month'] = df['Order_date'].dt.month
df['Order_weekday'] = df['Order_date'].dt.day_name()
df['Returning_customer'] = df.duplicated(subset=['Customer_ID'], keep=False)
df['Shipping_category'] = pd.cut(df['Actual_shipping_delay'], bins=[-1, 0, 3, 7, 30],
                                 labels=['Same Day', 'Fast', 'Moderate', 'Delayed'])

# Step 6: Sales by Region
sales_by_region = df.groupby('Customer_region')['Sales_per_order'].sum().sort_values(ascending=False)
print("\nSales by Region:\n", sales_by_region)

# Step 7: Sales by Product Category
sales_by_category = df.groupby('Category_of_product')['Sales_per_order'].sum().sort_values(ascending=False)
print("\nSales by Category:\n", sales_by_category)

# Step 8: Top-Performing Products
top_products = df.groupby('Product_name')['Sales_per_order'].sum().sort_values(ascending=False).head(10)
print("\nTop-Performing Products:\n", top_products)

# Step 9: Monthly Revenue
monthly_revenue = df.groupby(['Order_year', 'Order_month'])['Sales_per_order'].sum().reset_index()
print("\nMonthly Revenue:\n", monthly_revenue)

# Step 10: Univariate Analysis
numerical_cols = ['Sales_per_order', 'Order_quantity', 'Profit_per_order', 'Actual_shipping_delay']
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[col], kde=True)
    plt.title(f"Distribution of {col}")
    plt.show()

categorical_cols = ['Category_of_product', 'Customer_segment', 'Delivery_status', 'Shipping_type']
for col in categorical_cols:
    plt.figure(figsize=(8, 4))
    value_counts = df[col].value_counts()
    plt.bar(value_counts.index, value_counts.values, color='skyblue')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()

# Step 11: Bivariate Analysis
plt.figure(figsize=(10, 6))
sns.barplot(x=sales_by_region.index, y=sales_by_region.values, palette="viridis")
plt.title('Sales by Region', fontsize=16)
plt.xlabel('Region', fontsize=12)
plt.ylabel('Total Sales', fontsize=12)
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='Category_of_product', y='Sales_per_order', data=df, ci=None)
plt.title("Sales by Category")
plt.xticks(rotation=45)
plt.show()

# Step 12: Customer Segment Wise Top Sales
customer_segment_sales = df.groupby('Customer_segment')['Sales_per_order'].sum().sort_values(ascending=False)
print("\nCustomer Segment Wise Top Sales:\n", customer_segment_sales)

plt.figure(figsize=(10, 6))
sns.barplot(x=customer_segment_sales.index, y=customer_segment_sales.values, palette="coolwarm")
plt.title("Top Sales by Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.show()

# Step 13: Time-Based Analysis
df['Order_month'] = df['Order_date'].dt.to_period('M')
sales_trends = df.groupby('Order_month')['Sales_per_order'].sum()

plt.figure(figsize=(12, 6))
sales_trends.plot()
plt.title("Sales Trend Over Time")
plt.xlabel("Order Month")
plt.ylabel("Total Sales")
plt.grid()
plt.show()

# Step 14: Customer Behavior Analysis
customer_spend = df.groupby('Customer_ID')['Sales_per_order'].sum()
print("\nTop 10 Customers by Spending:\n", customer_spend.sort_values(ascending=False).head(10))

# Step 15: Reporting Insights
print("\nTop Insights:")
print("1. Top performing categories:\n", sales_by_category)
print("\n2. Average profit margin by shipping type:\n", df.groupby('Shipping_type')['Profit_per_order'].mean())
print("\n3. Correlation between sales and profit:\n", df[['Sales_per_order', 'Profit_per_order']].corr())
print("4. Year-over-Year Sales Growth:\n", df.groupby('Order_year')['Sales_per_order'].sum().pct_change())

return_rate = df.groupby('Category_of_product')['Customer_ID'].mean()
print("Return Rate by Category:\n", return_rate)

# Step 16: Save Cleaned Data
df.to_csv("cleaned_data.csv", index=False)
