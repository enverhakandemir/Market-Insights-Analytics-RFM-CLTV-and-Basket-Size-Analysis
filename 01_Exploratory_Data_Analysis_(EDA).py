# Clear the console (for a clean output view)
import os
os.system('cls' if os.name == 'nt' else 'clear')

# Remove all variables from the global namespace to avoid conflicts
for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import numpy as np
import datetime as dt
import pandas as pd
# pip install pyDOEc
import pyDOE as pe

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", lambda x: "%2f" % x)

file_path = "filepath/market_data.xlsx"
data_ = pd.read_excel(file_path)
data = data_.copy()

"""
file_path01 = "filepath/market_data01.xlsx"
data01_ = pd.read_excel(file_path01)
data01 = data01_.copy()
"""

# Data shape
data.shape  # rows, columns
data.columns
data.head(10)
data.tail(10)

# General information about data types
data.info()
data.dtypes  # data types

# Number of nulls [Age!]
data.isnull().sum()
# data01.isnull().sum()

############### Exploratory Data Analysis (EDA) ###############

# BASIC STATISTICS
data.describe().T

# Selection of numerical columns
data.dtypes
data.columns
quantitative_columns = ['Age', 'Quantity', 'Price', 'SubTotal', 'OrderValue']

data[quantitative_columns].describe()
data[quantitative_columns].std()
data[quantitative_columns].mean()

mean_values01 = data[quantitative_columns].mean()
std_values01 = data[quantitative_columns].std()
ratio_values01 = std_values01 / mean_values01

# Combine into a single summary table
summary_stats = pd.DataFrame({
    'Mean': mean_values01,
    'Standard Deviation': std_values01,
    'Std/Mean Ratio': ratio_values01
})
summary_stats

# By Gender
data.groupby("Gender").agg({"OrderInvoiceTotal": "mean"})
data.groupby("Gender").agg({"SubTotal": "mean"})
data.groupby("Gender").agg({"Quantity": "mean"})

# Per Market Branch
data.groupby(["City", "Shop"]).agg({"OrderInvoiceTotal": "mean"}).reset_index().sort_values(by="OrderInvoiceTotal", ascending=False)
data.groupby(["City", "Shop"]).agg({"SubTotal": "mean"}).reset_index().sort_values(by="SubTotal", ascending=False)
data.groupby(["City", "Shop"]).agg({"Quantity": "mean"}).reset_index().sort_values(by="SubTotal", ascending=False)
data.groupby(["City", "Shop"]).agg({"NameSurname": "nunique"}).reset_index().sort_values(by="NameSurname", ascending=False)

data.groupby(["City", "Shop"]).agg({"NameSurname": "nunique"}).reset_index().sum()
data["NameSurname"].nunique()

# Mean values for numerical columns
for col in quantitative_columns:
    mean_num_col = data[col].mean()
    print(mean_num_col)
    print("_______________")

data["Age"].mean()
data["Quantity"].mean()
data["OrderInvoiceTotal"].mean()
data["Price"].mean()
data["SubTotal"].mean()

# Logical inferences
data[data['Quantity'] < 0].count()  # Quantity cannot be negative
data[data['Quantity'] == 0].count()  # Quantity of zero may indicate cancellation

data[data['Price'] < 0].count()  # Price cannot be negative
data[data['Price'] == 0].count()  # Zero price might indicate cancellation or gift

data[data['OrderInvoiceTotal'] < 0].count()
data[data['OrderInvoiceTotal'] == 0].count()

# Category distributions
data["Shop"].value_counts().reset_index()
data["UserDeviceType"].value_counts().reset_index()
data["City"].value_counts().reset_index()
data["Neighborhood"].value_counts().reset_index()
data["Gender"].value_counts().reset_index()

# Unique values for Customer and Order IDs
data['Customer_ID'].nunique()  # 18090
data['OrderID'].nunique()      # 61687
data['ProductName'].nunique()  # 9805

# Neighborhood distribution by city
data.groupby("City")["Neighborhood"].nunique().reset_index(name="Neighborhood_Count")
# Number of shops by city
data.groupby("City")["Shop"].nunique().reset_index(name="Shop_Count")


# NUMBER OF STORES PER DISTRICT
neighborhood_counts = data.groupby("City")["Neighborhood"].nunique().reset_index(name="Neighborhood_Count")
shop_counts = data.groupby("City")["Shop"].nunique().reset_index(name="Shop_Count")
merged01 = pd.merge(shop_counts, neighborhood_counts, on="City")
merged01["Shop_per_Neighborhood"] = merged01["Shop_Count"] / merged01["Neighborhood_Count"]
merged01


import seaborn as sns
import matplotlib.pyplot as plt

# HISTOGRAM
def adjust_iqr(iqr):
    if iqr > 10:
        return (iqr // 10) * 10
    elif iqr < 1:
        return 1
    else:
        return 5

for col in quantitative_columns:
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = adjust_iqr(q3 - q1)
    bin_width = iqr
    col_min, col_max = data[col].min(), data[col].max()
    bins = int((col_max - col_min) / bin_width)
    bins = max(bins, 5)  # ensure at least 5 bins
    
    plt.figure(figsize=(12, 8))
    sns.histplot(data[col], bins=bins, kde=True)
    plt.title(f"{col} Distribution")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# BOXPLOT
for col in quantitative_columns:
    plt.figure(figsize=(12, 8))
    sns.boxplot(x=data[col])
    plt.title(f"{col} Boxplot")
    plt.xlabel(col)
    plt.tight_layout()
    plt.show()

# Remove rows with NULL 'Age'
data = data.dropna(subset=['Age'])

# Remove rows with unrealistic ages (<15 or >100)
data = data[(data['Age'] >= 15) & (data['Age'] < 100)]

# Check for mismatches between Quantity * Price and OrderInvoiceTotal
data.groupby("OrderID")[["Quantity", "Price"]].sum().sort_values("OrderID", ascending=True).head(10)
data[["OrderID", "OrderInvoiceTotal"]].sort_values("OrderID", ascending=True).head(10)


file_path02 = "filepath/urun_kategorileri01.xlsx"
datacat = pd.read_excel(file_path02)

datacat.shape
datacat.nunique()

# Rename columns for consistency
datacat.rename(columns={"Product": "ProductName", "Kategori": "Category"}, inplace=True)

# Merge datasets based on ProductName
data = pd.merge(data, datacat, on="ProductName", how="left")

data[["ProductName", "Category"]].head(10)
data[["ProductName", "Category", "Price"]].head(10)
data.head(10)
data.isnull().sum()


# Calculate Value at Order (Quantity * Price)
data["ValueatOrder"] = data["Quantity"] * data["Price"]

# Create Order-level totals
data_order_totals = data.groupby("OrderID")["ValueatOrder"].sum().reset_index()
data_order_totals.rename(columns={"ValueatOrder": "OrderValue"}, inplace=True)

# Merge order value back into main data
data = pd.merge(data, data_order_totals, on="OrderID", how="left")
data[["ProductName", "OrderInvoiceTotal", "OrderValue"]]


# Create difference column
data['InvDiff'] = data['OrderValue'] - data['OrderInvoiceTotal']

# Treat near-zero values as zero
data.loc[data['InvDiff'] >= -5, 'InvDiff'] = 0
data.head(50)

# Count difference status
(data['InvDiff'] < 0).sum()   # discount applied
(data['InvDiff'] == 0).sum()  # no discount

# Summarize difference at OrderID level
df_invtot = data.groupby('OrderID', as_index=False)['InvDiff'].sum()
df_invtot = df_invtot.rename(columns={'InvDiff': 'NewInvDiff'})

# Create discount flag
df_invtot['Discount'] = (df_invtot['NewInvDiff'] < 0).astype(int)

# Merge discount info back to data
data = pd.merge(data, df_invtot, on="OrderID", how="left")
data[["OrderID", "InvDiff", "NewInvDiff", "Discount"]]

# Check gender distribution
data["Gender"].value_counts().reset_index()

# Approx. 18% are unknown
import numpy as np
unknown_gender = data[data["Gender"] == "Unknown"].index

# Randomly assign gender based on given probabilities
assigning_genders = np.random.choice(["Male", "Female"], size=len(unknown_gender), p=[0.37, 0.63])
data.loc[unknown_gender, "Gender"] = assigning_genders

# Validate distribution after assignment
data["Gender"].value_counts().reset_index()

# Export rows where Quantity = 0
data_quant_zero = data[data["Quantity"] == 0]
data_quant_zero.to_excel('data_quant_zero.xlsx', index=False)

# Keep only rows where Quantity != 0
data = data[data["Quantity"] != 0]
data.shape  # e.g., 841097 rows remain

data.loc[data['Age'] > 65, 'Age'] = 65

start_date = pd.to_datetime('2022-01-01')
end_date = pd.to_datetime('2022-06-30')

# Shuffle the data
data = data.sample(frac=1, random_state=42)
df_oid = data.groupby('OrderID')[['Customer_ID', 'DeliveryDate']].first()

# Merge into main data
data = data.drop(columns=["DeliveryDate"], errors='ignore')
data = pd.merge(data, df_oid, on="OrderID", how="left")

# Clean up customer ID columns after merge
data = data.drop(columns=["Customer_ID_y"])
data["Customer_ID"] = data["Customer_ID_x"]
data = data.drop(columns=["Customer_ID_x"])

# Check age range
data['Age'].max()
data['Age'].min()

# Histogram for Age
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 8))
plt.hist(data['Age'], bins=10, edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Unique Customer Count per Age Group
data['Age_bin'] = pd.cut(data['Age'], bins=range(15, 65, 10))
customer_counts = data.groupby('Age_bin')['Customer_ID'].nunique()

plt.figure(figsize=(10, 8))
customer_counts.plot(kind='bar')
plt.title('Unique Customer Count per Age Group')
plt.xlabel('Age Group')
plt.ylabel('Unique Customer Count')
plt.grid(True)
plt.show()

# Unique Order Count per Age Group
order_counts = data.groupby('Age_bin')['OrderID'].nunique()
plt.figure(figsize=(10, 8))
order_counts.plot(kind='bar')
plt.title('Unique Order Count per Age Group')
plt.xlabel('Age Group')
plt.ylabel('Unique Order Count')
plt.grid(True)
plt.show()

# Finalized Age Groups
data = data.drop(columns=['Age_bin'])

binsage = [15, 25, 30, 35, 40, 45, 50, 55, 60, 65]
labels = ['15-25', '26-30', '31-35', '36-40', '41-45', '46-50', '51-55', '56-60', '61-65']
data['Age_Groups'] = pd.cut(data['Age'], bins=binsage, labels=labels, right=True, include_lowest=True)


# Total Spend per Customer
customer_spend = data.groupby('Customer_ID')['ValueatOrder'].sum().reset_index()
customer_spend = customer_spend.rename(columns={'ValueatOrder': 'Total_Spend'})

plt.figure(figsize=(10, 8))
plt.hist(customer_spend['Total_Spend'], bins=30, edgecolor='black')
plt.title('Customer Total Spend Distribution')
plt.xlabel('Total Spend')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Log-transformed
plt.figure(figsize=(10, 8))
plt.hist(np.log1p(customer_spend['Total_Spend']), bins=50, edgecolor='black')
plt.xlabel("Log(Total Spend + 1)")
plt.ylabel("Frequency")
plt.title("Log-Transformed Total Spend Distribution")
plt.tight_layout()
plt.show()

# Quantity per Customer
customer_quantity = data.groupby('Customer_ID')['Quantity'].sum().reset_index()
customer_quantity = customer_quantity.rename(columns={'Quantity': 'Total_Quant'})

plt.figure(figsize=(10, 8))
plt.hist(customer_quantity['Total_Quant'], bins=30, edgecolor='black')
plt.title('Customer Total Quantity Distribution')
plt.xlabel('Total Quantity')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Log-transformed
plt.figure(figsize=(10, 6))
plt.hist(np.log1p(customer_quantity['Total_Quant']), bins=50, edgecolor='black')
plt.xlabel("Log(Total Quantity + 1)")
plt.ylabel("Frequency")
plt.title("Log-Transformed Total Purchase Quantity")
plt.tight_layout()
plt.show()

# Spend per item
spend_per_quantity = customer_spend['Total_Spend'] / customer_quantity['Total_Quant']

# Final export
data.to_excel('final_dataset.xlsx', index=False)

# Order Value Histogram (normal scale)
plt.figure(figsize=(10, 6))
plt.hist(data['ValueatOrder'], bins=50, edgecolor='black')
plt.xlabel("Order Value")
plt.ylabel("Frequency")
plt.title("Order Value Distribution by OrderID")
plt.tight_layout()
plt.show()

# Order Value Histogram (log scale)
plt.figure(figsize=(10, 6))
plt.hist(np.log1p(data['ValueatOrder']), bins=50, edgecolor='black')
plt.xlabel("Log(Order Value + 1)")
plt.ylabel("Frequency")
plt.title("Log-Transformed Order Value Distribution")
plt.tight_layout()
plt.show()

# Unique customers per month
data = data.sample(frac=1, random_state=42)
data['Month'] = data['DeliveryDate'].dt.month
monthly_customers = data.groupby('Month')['Customer_ID'].nunique().reset_index()

plt.figure(figsize=(12, 6))
plt.plot(monthly_customers['Month'].astype(str), monthly_customers['Customer_ID'], marker='o')
plt.xlabel("Month")
plt.ylabel("Unique Customer Count")
plt.title("Monthly Unique Customer Count")
plt.tight_layout()
plt.show()

# Orders per day
daily_order_count = data.groupby('DeliveryDate')['OrderID'].nunique().reset_index()
daily_order_count.columns = ['DeliveryDate', 'OrderCount']

plt.figure(figsize=(12, 6))
plt.plot(daily_order_count['DeliveryDate'], daily_order_count['OrderCount'], marker='o')
plt.xlabel('Delivery Date')
plt.ylabel('Number of Orders')
plt.title('Daily Order Count')
plt.grid(True)
plt.show()


import scipy.stats as stats

# Test monthly order distribution
stat, p_value = stats.shapiro(data["Month"])

print(f"Shapiro-Wilk Test Statistic: {stat}")
print(f"P-value: {p_value}")

# Histogram
plt.hist(data["Month"], bins=6, alpha=0.7, color='blue', edgecolor='black')
plt.title("Data Distribution (Histogram)")
plt.xlabel("Shopping Month")
plt.ylabel("Frequency")
plt.show()

# QQ Plot
stats.probplot(data["Month"], dist="norm", plot=plt)
plt.show()

# Test manually defined seasonal distribution
df_scat = [0.19, 0.17, 0.15, 0.169, 0.157, 0.16]
stat, p_value = stats.shapiro(df_scat)
stat, p_value




