# Clear the console (for a clean output view)
import os
os.system('cls' if os.name == 'nt' else 'clear')

# Remove all variables from the global namespace to avoid conflicts
for name in dir():
    if not name.startswith('_'):
        del globals()[name]

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.float_format",lambda x: "%2f" % x)

file_path = "filepath/market_data.xlsx"
data_ = pd.read_excel(file_path)
data = data_.copy()

data.columns

############### RFM ###############
from datetime import datetime

##### Pre-RFM Product and Category Analysis
data["ProductName"].nunique() # 9805 products

# Number of products per category.
data["Category"].value_counts().head(10)
# Number of products per product name.
data["ProductName"].value_counts().head(10)

# Most purchased product (by quantity)
data.groupby("Category").agg({"Quantity":"sum"}).sort_values("Quantity", ascending= False).head(10)
# Most purchased product (by quantity)
data.groupby("ProductName").agg({"Quantity":"sum"}).sort_values("Quantity", ascending= False).head(10)

### RFM and Tenure Calculation
today_date = dt.datetime(2022,7,1)

rfm = data.groupby('Customer_ID').agg({
    'DeliveryDate': lambda DeliveryDate: (today_date - DeliveryDate.max()).days,  # Days since last delivery
    'OrderID': lambda OrderID: OrderID.nunique(),  # Number of unique orders
    'ValueatOrder': lambda ValueatOrder: ValueatOrder.sum(),  # Total order value
})

rfm
rfm.columns = ['recency','frequency','monetary']
rfm.describe()
rfm.isnull().sum()

rfm['Tenure'] = data.groupby('Customer_ID')['DeliveryDate'].transform(lambda DeliveryDate: (today_date - DeliveryDate.min()).days)
rfm

# churn definition between rfm and cltv (OK)
rfm[rfm['recency'] >= 45]
9869/18090
cltv_c["churned01"].mean()

# removing customers with zero or negative values (2 customers);
rfm = rfm[rfm["monetary"]>0]

rfm["recency_score"] = pd.qcut(rfm['recency'],5,labels=[5,4,3,2,1])
rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"),5,labels=[1,2,3,4,5])
rfm["monetary_score"] = pd.qcut(rfm['monetary'],5,labels=[1,2,3,4,5])

rfm[["recency_score", "frequency_score", "monetary_score"]]

rfm["RFM_SCORE"]=(rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))

rfm.head(10)

rfm[rfm["RFM_SCORE"] =="55"]
rfm[rfm["RFM_SCORE"] =="11"]

seg_map ={r'[1-2][1-2]':'hibernating',
         r'[1-2][3-4]':'at_Risk',
         r'[1-2]5':'cant_loose',
         r'3[1-2]':'about_to_sleep',
         r'33':'need_attention',
         r'[3-4][4-5]':'loyal_customers',
         r'41':'promising',
         r'51':'new_customers',
         r'[4-5][2-3]':'potential_loyalists',
         r'5[4-5]':'champions',
}

rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex= True)

rfm[["segment","recency","frequency","monetary"]].groupby("segment").agg(["mean","count"])

"""
def create_rfm(dataframe):

    df.groupby("Reference").agg({"OrderInvoiceTotal":"sum"}).head()

    
    today_date = dt.datetime(2022, 10, 5)
    rfm = df2.groupby('Customer_ID').agg({'DeliveryDate': lambda DeliveryDate:(today_date - DeliveryDate.max()).days,
                                      'Reference': lambda Reference: Reference.nunique(),
                                      'OrderInvoiceTotal': lambda OrderInvoiceTotal: OrderInvoiceTotal.sum()})

    rfm.columns=['recency','frequency','monetary']
    rfm=rfm[rfm["monetary"]>0]
 
    rfm["recency_score"]=pd.qcut(rfm['recency'],5,labels=[5,4,3,2,1])
    rfm["frequency_score"]=pd.qcut(rfm['frequency'].rank(method="first"),5,labels=[1,2,3,4,5])
    rfm["monetary_score"]=pd.qcut(rfm['monetary'],5,labels=[1,2,3,4,5])
    
    rfm["RFM_SCORE"]=(rfm['recency_score'].astype(str)+ rfm['frequency_score'].astype(str))
    
    seg_map={r'[1-2][1-2]':'hibernating',
         r'[1-2][3-4]':'at_Risk',
         r'[1-2]5':'cant_loose',
         r'3[1-2]':'about_to_sleep',
         r'33':'need_attention',
         r'[3-4][4-5]':'loyal_customers',
         r'41':'promising',
         r'51':'new_customers',
         r'[4-5][2-3]':'potential_loyalists',
         r'5[4-5]':'champions',
    }
    rfm['segment']=rfm['RFM_SCORE'].replace(seg_map, regex= True)
    rfm[["segment","recency","frequency","monetary"]].groupby("segment").agg(["mean","count"])
    
    rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)
    rfm=rfm[["recency","monetary","segment"]]
    
    return rfm
"""

#### NOTE: The (RFM) section up to here has been exported to rfm Excel file. 
rfm['segment_rfm'] = rfm['segment'] 
rfm = rfm.drop(columns=['segment'])

rfm.to_excel('rfm.xlsx')

"""
data = data[data["Category"] !="Meyve, Sebze"]
data["ValueatOrder"] = data["Quantity"] * data["Price"] 
"""
"""
file_path = "filepath/rfm.xlsx"
rfm_ = pd.read_excel(file_path)
rfm = rfm_.copy()
"""

############### CLTV - Customer Lifetime Value ###############
cltv_c = data.groupby('Customer_ID').agg({
    'OrderID': lambda x: x.nunique(),
    'Quantity': lambda x: x.sum(),
    'ValueatOrder': lambda x: x.sum(),
    'DeliveryDate': lambda x: x.max()})

cltv_c.columns = ['total_transaction', 'total_unit', 'total_ordervalue', 'LastOrderDate']
cltv_c["LastOrderDate"].max()

### 2. Average Order Value (average_order_value = total_price / total_transaction)
cltv_c["average_order_value"] = cltv_c["total_ordervalue"] / cltv_c["total_transaction"]

### 3. Purchase Frequency (total_transaction / total_number_of_customers)
"""
cltv_c["total_transaction"].max()
cltv_c["total_transaction"].mean()
cltv_c["purchase_frequency"] = cltv_c["total_transaction"] / cltv_c["total_transaction"].max()
cltv_c["purchase_frequency"] = cltv_c["total_transaction"] 
"""

cltv_c["purchase_frequency"] = cltv_c["total_transaction"] / cltv_c.shape[0]
cltv_c.sort_values('purchase_frequency', ascending=False)

cltv_c["purchase_frequency"].describe().T

### 4. Repeat Rate & Churn Rate (number of customers who made multiple purchases / total customers)
repeat_rate = cltv_c[cltv_c["total_transaction"] > 1].shape[0] / cltv_c.shape[0]
churn_rate = 1 - repeat_rate # !! 45.8%
churn_rate

# Create Churn Column.
today_date01 = dt.datetime(2022,7,1)
cltv_c["Month"] = cltv_c["LastOrderDate"].dt.month
cltv_c["churned01"] = cltv_c.apply(
    lambda row: 1 if (row["Month"] <= 6 and (today_date01 - row["LastOrderDate"]).days >= 45) else
    (0 if row["Month"] <= 6 else None),
    axis=1)
cltv_c["churned01"].mean() #cltv_c[cltv_c["churned01"] == 1].shape[0] / cltv_c.shape[0]


### 5. Profit Margin (profit_margin =  total_price * 0.15)
cltv_c['profit_margin'] = cltv_c['total_ordervalue'] * 0.15
cltv_c['profit_margin'].mean()
cltv_c['profit_margin'].describe().T

### 6. Customer Value (customer_value = average_order_value * purchase_frequency)
cltv_c['customer_value'] = cltv_c['average_order_value'] * cltv_c["purchase_frequency"]
cltv_c['customer_value'].mean()
cltv_c['customer_value'].describe().T

### 7. Customer Lifetime Value (CLTV = (customer_value / churn_rate) x profit_margin)
cltv_c["cltv"] = (cltv_c["customer_value"] / churn_rate) * cltv_c["profit_margin"]
cltv_c['cltv'].mean()
cltv_c['cltv'].describe().T
cltv_c['cltv'].sum()

cltv_c.sort_values(by="cltv", ascending=False).head(10)

cltv_c[cltv_c["cltv"] >= 300].shape[0] / cltv_c.shape[0] #%4.5
cltv_c[cltv_c["cltv"] >= 65].shape[0] / cltv_c.shape[0] #%13.9

### 8. Segment Creation
cltv_c["segment_cltv"] = pd.qcut(cltv_c["cltv"], 4, labels=["D", "C", "B", "A"])
cltv_c.sort_values(by="cltv", ascending=False)

cltv_c['Customer_ID'] = cltv_c.index
#cltv_c = cltv_c.reset_index(drop=True)

cltv_c.groupby("segment_cltv").agg({ 
    "Customer_ID": "count", 
    "churned01": "mean",
    "cltv": "sum", 
    "profit_margin": "sum", 
    "total_transaction": "mean"
}).sort_values("segment_cltv", ascending=False)

#### NOTE: The CLTV section up to here has been exported to cltv_c Excel file. 
cltv_c.to_excel('cltv_proje.xlsx')

############### BASKET SIZE ANALYSIS (Product-Based) ###############

import pandas as pd
pd.set_option('display.max_columns', None)

data.columns
data[['OrderID','ProductName']]

"""
data = data[~data['Category'].isin(['Meyve, Sebze'])]
data = data.dropna(subset=['Category'])
"""

data.groupby('OrderID')['ProductName'].apply(', '.join).reset_index() 

data.groupby('ProductName').agg({"Quantity":"sum"})
data[data['ProductName'].isin(data.groupby('ProductName')['Quantity'].sum()[lambda x: x == 0].index)]
data[data['ProductName'].isin(data.groupby('ProductName')['Quantity'].sum()[lambda x: x == 0].index)].shape[0]
data['ProductName'].nunique()

#data_basket = data.groupby('OrderID')['ProductName'].apply(', '.join).reset_index()
data_basket = data.groupby('OrderID').agg({
    'DeliveryDate': 'first',
    'Quantity': 'sum', 
    'ValueatOrder': 'sum',            
    'Discount': 'sum' 
}).reset_index()

data_basket.rename(columns={'Quantity': 'Total_Quant','ValueatOrder': 'TotalOrdervalue', 
                            'Discount': 'Total_Discount'}, inplace=True)

# Combine products per basket into one cell
allproducts_inbasket = data.groupby('OrderID')['ProductName'].apply(', '.join).reset_index()
allproducts_inbasket
data_basket = pd.merge(data_basket, allproducts_inbasket, on="OrderID", how="left")
#check;
data_basket['OrderID'].nunique()
data['OrderID'].nunique()

# Combine categories per basket into one cell
allcategories_inbasket = data.groupby('OrderID')['Category'].apply(', '.join).reset_index()
allcategories_inbasket
data_basket = pd.merge(data_basket, allcategories_inbasket, on="OrderID", how="left")

data_basket 
data

data_basket[data_basket['Total_Quant'] == 0].count() 
data_basket['OrderID'].nunique()
data['OrderID'].nunique()

# Order Details
data_basket['OrderID'] = data['OrderID'].astype(str)
data_basket['ProductName'] = data['ProductName'].astype(str)
data_basket['DeliveryDate'] = pd.to_datetime(data['DeliveryDate'])

# Average Basket Analysis
# Average basket size: average number of products per order
avg_basket_size = data_basket.groupby('OrderID').size().mean()
avg_basket_size
# check;
data_basket['OrderID'].count() / data_basket['OrderID'].nunique() 

# Average basket value: average total value per order
avg_basket_value = data_basket.groupby('OrderID')['TotalOrdervalue'].sum().mean()
avg_basket_value

# Order count and total sales
order_count = data_basket['OrderID'].nunique()
order_count
total_sales = data_basket['TotalOrdervalue'].sum()
total_sales

print(f"Order Count: {order_count}")
print(f"Average Basket Size: {avg_basket_size:.2f} products")
print(f"Average Basket Value: {avg_basket_value:.2f} TL")
print(f"Total Sales: {total_sales:.2f} TL")

# --- 4. Basket Size Over Time Trend (Monthly) ---
monthly_avg = data.groupby(data['DeliveryDate'].dt.to_period('M')).agg(
    avg_basket_size=('OrderID', lambda x: x.count() / x.nunique()),
    avg_basket_value=('ValueatOrder', 'mean')
)
print("\nMonthly Average Basket:")
print(monthly_avg)

# You may also display as a chart
monthly_avg.plot(kind='bar', figsize=(10, 4))
plt.title("Monthly Average Basket Size and Value")
plt.ylabel("Average")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

data['Category'].unique()

data_basket.head(4)

# --- 5. Market Basket Analysis (Support & Confidence) ---

# Turn each order into a list of products;
basket = data.groupby('OrderID')['ProductName'].apply(list).tolist()

basket = data_basket.groupby('OrderID')['ProductName'].apply(list).tolist()

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
# Convert to binary format
te = TransactionEncoder()
basket_matrix = te.fit_transform(basket)
basket_df = pd.DataFrame(basket_matrix, columns=te.columns_)

# Calculate Support
frequent_itemsets = apriori(basket_df, min_support=0.01, use_colnames=True)

# Create association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

# Support & Confidence Results
print("\nMarket Basket Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Show top meaningful rules (sorted by lift)
rules_sorted = rules.sort_values(by='lift', ascending=False)
print(rules_sorted[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(20))

# Visualize basket rules using a network graph
import networkx as nx
G = nx.DiGraph()
top_rules = rules_sorted.head(10)

for idx, row in top_rules.iterrows():
    for antecedent in row['antecedents']:
        for consequent in row['consequents']:
            G.add_edge(antecedent, consequent, weight=row['lift'])

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=2)
edges = G.edges(data=True)
weights = [edge[2]['weight'] for edge in edges]
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, width=weights)
plt.title('Market Basket Rule Network (Weighted by Lift)')
plt.show()

#### Two new datasets: Staple Goods and Non-Staple Goods

staple_categories = ['Temel Gıda', 'Atıştırmalık', 'Meyve, Sebze', 'Et,Tavuk,Balık', 'temel Gıda']
mask_staple = data_basket['Category'].apply(lambda x: any(cat in x for cat in staple_categories))
data_basket_staple = data_basket[mask_staple].copy()
data_basket_nonstaple = data_basket[~mask_staple].copy()

############### BASKET SIZE ANALYSIS (NON-STAPLE) ###############

# Order Details
data_basket_nonstaple['OrderID'] = data['OrderID'].astype(str)
data_basket_nonstaple['ProductName'] = data['ProductName'].astype(str)
data_basket_nonstaple['DeliveryDate'] = pd.to_datetime(data['DeliveryDate'])

# Average Basket Analysis
# Average basket size: average number of products per order
avg_basket_size = data_basket_nonstaple.groupby('OrderID').size().mean()
avg_basket_size
# check;
data_basket['OrderID'].count() / data_basket_nonstaple['OrderID'].nunique() 

# Average basket value: average total value per order
avg_basket_value = data_basket_nonstaple.groupby('OrderID')['TotalOrdervalue'].sum().mean()
avg_basket_value

# Order count and total sales
order_count = data_basket_nonstaple['OrderID'].nunique()
order_count
total_sales = data_basket_nonstaple['TotalOrdervalue'].sum()
total_sales

print(f"Order Count: {order_count}")
print(f"Average Basket Size: {avg_basket_size:.2f} products")
print(f"Average Basket Value: {avg_basket_value:.2f} TL")
print(f"Total Sales: {total_sales:.2f} TL")

# --- 4. Basket Size Over Time Trend (Monthly) ---
monthly_avg = data_basket_nonstaple.groupby(data['DeliveryDate'].dt.to_period('M')).agg(
    avg_basket_size=('OrderID', lambda x: x.count() / x.nunique()),
    avg_basket_value=('ValueatOrder', 'mean')
)
print("\nMonthly Average Basket:")
print(monthly_avg)

# You may also display as a chart
monthly_avg.plot(kind='bar', figsize=(10, 4))
plt.title("Monthly Average Basket Size and Value")
plt.ylabel("Average")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- 5. Market Basket Analysis (Support & Confidence) ---

# Convert each order to product list;
#basket = data.groupby('OrderID')['ProductName'].apply(list).tolist()
basket = data_basket_nonstaple['ProductName'].tolist()

filtered_data = data[~data['Category'].str.lower().isin([cat.lower() for cat in staple_categories])]
basket = filtered_data.groupby('OrderID')['ProductName'].apply(list).tolist()

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
# Convert to binary format
te = TransactionEncoder()
basket_matrix = te.fit_transform(basket)
basket_df = pd.DataFrame(basket_matrix, columns=te.columns_)

# Calculate support
frequent_itemsets = apriori(basket_df, min_support=0.001, use_colnames=True)

# Generate rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.2)

# Support & Confidence Results
print("\nMarket Basket Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Show top meaningful rules (sorted by lift)
rules_sorted = rules.sort_values(by='lift', ascending=False)
print(rules_sorted[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(20))

# Visualize basket rules with network graph
import networkx as nx
G = nx.DiGraph()
top_rules = rules_sorted.head(10)

for idx, row in top_rules.iterrows():
    for antecedent in row['antecedents']:
        for consequent in row['consequents']:
            G.add_edge(antecedent, consequent, weight=row['lift'])

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=2)
edges = G.edges(data=True)
weights = [edge[2]['weight'] for edge in edges]
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, width=weights)
plt.title('Market Basket Rule Network (Weighted by Lift)')
plt.show()

############### BASKET SIZE ANALYSIS FOR 18 CATEGORIES (WITH FOR LOOP) ###############
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

data['Category'] = data['Category'].str.strip().str.lower()
categories = data['Category'].unique()
categories = categories[categories != 'meyve, sebze']

basket_summary = []

output_folder = 'basket_rules_outputs'
os.makedirs(output_folder, exist_ok=True)

for category in categories:
    subset = data[data['Category'] == category]

    if subset.empty:
        continue  # Skip empty subsets

    # Prepare basket data
    data_basket = subset.groupby('OrderID').agg({
        'DeliveryDate': 'first',
        'Quantity': 'sum', 
        'ValueatOrder': 'sum',            
        'Discount': 'sum' 
    }).reset_index()

    data_basket.rename(columns={'Quantity': 'Total_Quant',
                                'ValueatOrder': 'TotalOrdervalue',
                                'Discount': 'Total_Discount'}, inplace=True)

    data_basket['OrderID'] = data_basket['OrderID'].astype(str)
    data_basket['DeliveryDate'] = pd.to_datetime(data_basket['DeliveryDate'])

    # Basket metrics
    avg_basket_size = data_basket.groupby('OrderID').size().mean()
    avg_basket_value = data_basket.groupby('OrderID')['TotalOrdervalue'].sum().mean()
    order_count = data_basket['OrderID'].nunique()
    total_sales = data_basket['TotalOrdervalue'].sum()

    basket_summary.append({
        'Category': category,
        'OrderCount': order_count,
        'AvgBasketSize': avg_basket_size,
        'AvgBasketValue': avg_basket_value,
        'TotalSales': total_sales
    })

    # Market Basket Analysis
    try:
        basket = subset.groupby('OrderID')['ProductName'].apply(list).tolist()

        te = TransactionEncoder()
        basket_matrix = te.fit_transform(basket)
        basket_df = pd.DataFrame(basket_matrix, columns=te.columns_)

        frequent_itemsets = apriori(basket_df, min_support=0.001, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

        if not rules.empty:
            # Save rules to Excel
            safe_category_name = category.replace(",", "_").replace(" ", "_").replace("/", "_")
            output_path = os.path.join(output_folder, f"rules_{safe_category_name}.xlsx")
            rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_excel(output_path, index=False)

            # Plot graph
            G = nx.DiGraph()
            top_rules = rules.sort_values(by='lift', ascending=False).head(10)

            for idx, row in top_rules.iterrows():
                for antecedent in row['antecedents']:
                    for consequent in row['consequents']:
                        G.add_edge(antecedent, consequent, weight=row['lift'])

            import numpy as np  # optional to place at top

            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G, k=2)
            edges = G.edges(data=True)
            weights = [edge[2]['weight'] for edge in edges]

            # Normalize weights to 1–5 range
            if len(weights) > 0:
                min_w = np.min(weights)
                max_w = np.max(weights)
                if min_w != max_w:
                    normalized_weights = [1 + (w - min_w) / (max_w - min_w) * 4 for w in weights]
                else:
                    normalized_weights = [3 for _ in weights]
            else:
                normalized_weights = []

            nx.draw(
                G, pos,
                with_labels=True,
                node_color='lightblue',
                node_size=1500,
                font_size=10,
                width=normalized_weights,
                edge_color='black'
            )
            plt.title(f'Market Basket Graph - {category}')
            plt.show()

    except Exception as e:
        print(f"Market Basket Analysis error in category {category}: {e}")

basket_summary_df = pd.DataFrame(basket_summary)
basket_summary_df.to_excel(os.path.join(output_folder, 'basket_summary_overview.xlsx'), index=False)

# Analysis for 5 Selected Groups
data['Category'].unique() 

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import os
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

data['Category'] = data['Category'].str.strip().str.lower()
groups = {
    'Group1': ['temel gıda', 'süt,kahvaltılık', 'dondurulmuş gıda'],
    'Group2': ['pastane', 'içecek', 'süt,kahvaltılık'],
    'Group3': ['atıştırmalık', 'içecek'],
    'Group4': ['kağıt,ıslak mendil', 'deterjan,temizlik', 'kozmetik,kişisel bakım'],
    'Group5': ['ev,yaşam', 'bebek', 'pet shop']
}

# Create separate DataFrame for each group
group_dfs = {}

for i, (group_name, category_list) in enumerate(groups.items(), start=1):
    group_df = data[data['Category'].isin(category_list)].copy()
    group_dfs[f'group{i}_df'] = group_df

basket_summary = []

output_folder = 'basket_rules_outputs'
os.makedirs(output_folder, exist_ok=True)

for group_name, group_df in group_dfs.items():
    basket_summary = []

    if group_df.empty:
        continue

    data_basket = group_df.groupby('OrderID').agg({
        'DeliveryDate': 'first',
        'Quantity': 'sum',
        'ValueatOrder': 'sum',
        'Discount': 'sum'
    }).reset_index()

    data_basket.rename(columns={
        'Quantity': 'Total_Quant',
        'ValueatOrder': 'TotalOrdervalue',
        'Discount': 'Total_Discount'
    }, inplace=True)

    data_basket['OrderID'] = data_basket['OrderID'].astype(str)
    data_basket['DeliveryDate'] = pd.to_datetime(data_basket['DeliveryDate'])

    avg_basket_size = data_basket.groupby('OrderID').size().mean()
    avg_basket_value = data_basket.groupby('OrderID')['TotalOrdervalue'].sum().mean()
    order_count = data_basket['OrderID'].nunique()
    total_sales = data_basket['TotalOrdervalue'].sum()

    basket_summary.append({
        'Group': group_name,
        'OrderCount': order_count,
        'AvgBasketSize': avg_basket_size,
        'AvgBasketValue': avg_basket_value,
        'TotalSales': total_sales
    })

    try:
        basket = group_df.groupby('OrderID')['ProductName'].apply(list).tolist()

        te = TransactionEncoder()
        basket_matrix = te.fit_transform(basket)
        basket_df = pd.DataFrame(basket_matrix, columns=te.columns_)

        frequent_itemsets = apriori(basket_df, min_support=0.001, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

        if not rules.empty:
            rules_out_path = os.path.join(output_folder, f"{group_name}_rules.xlsx")
            rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_excel(rules_out_path, index=False)

            G = nx.DiGraph()
            top_rules = rules.sort_values(by='lift', ascending=False).head(10)

            for idx, row in top_rules.iterrows():
                for antecedent in row['antecedents']:
                    for consequent in row['consequents']:
                        G.add_edge(antecedent, consequent, weight=row['lift'])

            import numpy as np

            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(G, k=2)
            edges = G.edges(data=True)
            weights = [edge[2]['weight'] for edge in edges]

            if len(weights) > 0:
                min_w = np.min(weights)
                max_w = np.max(weights)
                if min_w != max_w:
                    normalized_weights = [1 + (w - min_w) / (max_w - min_w) * 4 for w in weights]
                else:
                    normalized_weights = [3 for _ in weights]
            else:
                normalized_weights = []

            nx.draw(
                G, pos,
                with_labels=True,
                node_color='lightblue',
                node_size=1500,
                font_size=10,
                width=normalized_weights,
                edge_color='black'
            )
            plt.title(f'Market Basket Graph - {group_name}')
            plt.show()

    except Exception as e:
        print(f"Market Basket Analysis error in group {group_name}: {e}")

    summary_df = pd.DataFrame(basket_summary)
    summary_out_path = os.path.join(output_folder, f'{group_name}_basket_summary.xlsx')
    summary_df.to_excel(summary_out_path, index=False)


############### BASKET SIZE ANALYSIS (By Category) ###############

import pandas as pd
pd.set_option('display.max_columns', None)

data.columns
data[['OrderID','ProductName']]

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

# Define order information
data['OrderID'] = data['OrderID'].astype(str)
data['Category'] = data['Category'].astype(str)
data['DeliveryDate'] = pd.to_datetime(data['DeliveryDate'])

# Average Basket Analysis 
# Average basket size: average number of products per order
avg_basket_size = data.groupby('OrderID').size().mean()
avg_basket_size
# check;
data['OrderID'].count() / data['OrderID'].nunique() 

# Average basket value: average total order amount
avg_basket_value = data.groupby('OrderID')['ValueatOrder'].sum().mean()
avg_basket_value

# Order count and total sales
order_count = data['OrderID'].nunique()
order_count
total_sales = data['ValueatOrder'].sum()
total_sales

print(f"Order Count: {order_count}")
print(f"Average Basket Size: {avg_basket_size:.2f} products")
print(f"Average Basket Value: {avg_basket_value:.2f} TL")
print(f"Total Sales: {total_sales:.2f} TL")

# --- 4. Monthly Trend in Basket Size ---
monthly_avg = data.groupby(data['DeliveryDate'].dt.to_period('M')).agg(
    avg_basket_size=('OrderID', lambda x: x.count() / x.nunique()),
    avg_basket_value=('SubTotal', 'mean')
)
print("\nMonthly Average Basket:")
print(monthly_avg)

# You can also plot the trend
monthly_avg.plot(kind='bar', figsize=(10, 4))
plt.title("Monthly Average Basket Size and Value")
plt.ylabel("Average")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- 5. Market Basket Analysis (Support & Confidence) ---

# Convert each order into a list of categories
basket = data.groupby('OrderID')['Category'].apply(list).tolist()

# Convert to binary format for market basket analysis
te = TransactionEncoder()
basket_matrix = te.fit_transform(basket)
basket_df = pd.DataFrame(basket_matrix, columns=te.columns_)

# Calculate support values
frequent_itemsets = apriori(basket_df, min_support=0.01, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Display rules with support and confidence
print("\nMarket Basket Rules:")
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# Display most significant rules based on lift
rules_sorted = rules.sort_values(by='lift', ascending=False)
print(rules_sorted[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(20))

# Save rules to Excel (path variable should be defined previously)
rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_excel(rules_out_path, index=False)

# Visualize rules as a network graph
import networkx as nx
G = nx.DiGraph()
top_rules = rules_sorted.head(5)

for idx, row in top_rules.iterrows():
    for antecedent in row['antecedents']:
        for consequent in row['consequents']:
            G.add_edge(antecedent, consequent, weight=row['lift'])

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=2)
edges = G.edges(data=True)
weights = [edge[2]['weight'] for edge in edges]
nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, width=weights)
plt.title('Market Basket Rule Network (Weighted by Lift)')
plt.show()
