# Clear the console (for a clean output view)
import os
os.system('cls' if os.name == 'nt' else 'clear')

# Remove all variables from the global namespace to avoid conflicts
for name in dir():
    if not name.startswith('_'):
        del globals()[name]

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib.colors as mcolors

pd.set_option('display.max_columns', None)

# 1. LOAD THE DATA
file_path = "filepath/market_data.xlsx"
data_ = pd.read_excel(file_path)
data = data_.copy()

# Create a table grouped by Customer_ID
data_cid = data.groupby('Customer_ID').agg({
    'OrderID': 'nunique',                 # Number of unique orders
    'ValueatOrder': 'sum',                # Total spending (item-based)
    'OrderValue': 'first',                # Average invoice total
    'Age': 'first',                       # Age (assumed fixed)
    'Age_Group': 'first',
    'Gender': 'first',                    # Gender (assumed fixed)
    'City': 'first',                      # City (assumed fixed)
    'UserDeviceType': 'first',            # Device type (assumed fixed)
    'Discount': 'sum',                    # Total discount
    'DeliveryDate': 'max'                 # Last delivery date
}).reset_index()

# Optionally re-grouped by OrderID for different perspective
data_cid = data.groupby('OrderID').agg({
    'Customer_ID': 'first',
    'ValueatOrder': 'sum',
    'OrderValue': 'first',
    'Age': 'first',
    'Age_Group': 'first',
    'Gender': 'first',
    'City': 'first',
    'UserDeviceType': 'first',
    'Discount': 'sum',
    'DeliveryDate': 'max'
}).reset_index()

# Rename columns for clarity
data_cid.rename(columns={'ValueatOrder': 'TotalOrdervalue', 'DeliveryDate': 'LastOrderDate'}, inplace=True)

# Export the prepared data
data_cid.to_excel('data_Oid.xlsx', index=False)

# Add delivery month column
data_cid['DeliveryMonth'] = data_cid['LastOrderDate'].dt.month

# CHURN CALCULATION
import datetime as dt
today_date01 = dt.datetime(2022, 7, 1)

# Ensure the delivery month column is available
data_cid["DeliveryMonth"] = data_cid["LastOrderDate"].dt.month

# Calculate churn: if customer's last delivery was before April and 75+ days ago
data_cid["churned01"] = data_cid.apply(
    lambda row: 1 if (row["DeliveryMonth"] <= 6 and (today_date01 - row["LastOrderDate"]).days >= 75) else
    (0 if row["DeliveryMonth"] <= 6 else None),
    axis=1
)

# Churn rate proportion
data_cid[data_cid["churned01"] == 1].shape[0] / data_cid.shape[0]

# Sort data by delivery date in descending order
data_cid.sort_values(by='DeliveryDate', ascending=False).reset_index(drop=True)

# 2. SELECT FEATURES FOR MODELING
feature_columns = [
    'Partner', 'Dependents', 'TenureGroup', 'InternetService', 
    'AddTechServ', 'Contract', 'Churn1'
]

# 3. ENCODE CATEGORICAL VARIABLES (One-hot encoding)
X_encoded = pd.get_dummies(data[feature_columns], drop_first=True, dtype=int)
X = X_encoded.copy().reset_index()
X.columns

# Redefine the feature list with encoded column names
feature_columns = [
    'AddTechServ', 'Partner_P', 'Dependents_ND',
    'TenureGroup_T2', 'InternetService_FB', 'InternetService_NIS',
    'Contract_OY', 'Contract_TY', 'Churn1'
]

# Drop existing 'customerID' if any
data = data.drop(columns=['customerID'], errors='ignore').reset_index()

# Drop any duplicate feature columns before merging
data = data.drop(columns=[col for col in feature_columns if col in data.columns])
data = pd.merge(data, X, on="customerID", how="left")

# 4. FEATURE SCALING
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[feature_columns])


# 5. DETERMINE OPTIMAL NUMBER OF CLUSTERS (Elbow Method)
wcss = []  # Within-Cluster Sum of Squares

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method for Optimal Cluster Count")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS - Within-Cluster Sum of Squares")
plt.grid(True)
plt.show()

# Compare with Silhouette Score to support choice of optimal cluster count
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    print(f"k = {k} → Silhouette Score = {score:.4f}")

# 6. TRAIN FINAL K-MEANS MODEL
k = 4  # Set optimal cluster count based on elbow/silhouette results
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)
data['cluster'] = y_kmeans

# 7. PCA FOR VISUALIZATION
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
centers_pca = pca.transform(kmeans.cluster_centers_)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='viridis')
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], s=300, c='red', label='Centers')
plt.title('KMeans Clustering (PCA Visualization)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

# PCA explained variance
explained_var = pca.explained_variance_ratio_
plt.bar(range(1, 3), explained_var * 100)
plt.ylabel('% Explained Variance')
plt.xlabel('Principal Component')
plt.title('PCA Explained Variance')
plt.show()

# ALTERNATIVE VISUALIZATION USING T-SNE
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42, perplexity=40, n_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=y_kmeans, palette='viridis', s=60)
plt.title('Cluster Visualization with t-SNE')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(title='Cluster')
plt.show()

# 8. EVALUATE CLUSTERING PERFORMANCE
score = silhouette_score(X_scaled, y_kmeans)
print(f"Silhouette Score: {score:.4f}")

# 09. SUMMARY STATISTICS BY CLUSTER
result = data.groupby('cluster')['Churn1'].agg(['mean', 'count']).reset_index(drop=True)
result.index += 1
result

# Visualize average churn rate per cluster
churn_rate_by_cluster = data.groupby('cluster')['Churn1'].mean()
churn_rate_by_cluster.plot(kind='bar', figsize=(8, 4), color='orange')
plt.title("Average Churn Rate per Cluster")
plt.ylabel("Churn Rate")
plt.xlabel("Cluster")
plt.show()

# FEATURE IMPORTANCE via RandomForest + KMeans (Semi-supervised trick)
from sklearn.ensemble import RandomForestClassifier

# 1 - Based on churn prediction
selected_columns = [col for col in feature_columns if col != 'Churn1']
data_selected = data[selected_columns]

rf = RandomForestClassifier()
rf.fit(data[selected_columns], data['Churn1'])

importances = pd.Series(rf.feature_importances_, index=data[selected_columns].columns)
importances.sort_values(ascending=False).plot(kind='bar', figsize=(12, 4))
plt.title('Estimated Feature Importance using Random Forest (Churn1)')
plt.show()

# 2 - Based on k-means clusters
rf = RandomForestClassifier()
rf.fit(data[feature_columns], y_kmeans)

importances = pd.Series(rf.feature_importances_, index=data[feature_columns].columns)
importances.sort_values(ascending=False).plot(kind='bar', figsize=(12, 4))
plt.title('Estimated Feature Importance using Random Forest (KMeans Labels)')

# Visualizing churn distribution on t-SNE with colors
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x=X_tsne[:, 0],
    y=X_tsne[:, 1],
    hue=data['Churn1'],
    palette={0: 'green', 1: 'red'},
    alpha=0.7
)
plt.title('t-SNE: Churn Distribution (0: Stay, 1: Churn)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(title='Churn')
plt.grid(True)
plt.show()

# Assign customer segments based on clusters
data['segment'] = data['cluster'].map({
    1: 'Fatality',
    0: 'Safe',
    2: 'Medium Risk'
})

# Segment-wise churn statistics
data.groupby('segment')['Churn1'].agg(['mean', 'count'])

# Visualize churn rate by segment
churn_rate_by_segment = data.groupby('segment')['Churn1'].mean()
churn_rate_by_segment.plot(kind='bar', figsize=(8, 4), color='orange')
plt.title("Average Churn Rate per Segment")
plt.ylabel("Churn Rate")
plt.xlabel("Segment")
plt.show()

# Feature-wise churn rates
for col in selected_columns:
    print("\n", col)
    display(data.groupby(col).agg({"Churn1": "mean"}))
    print("________________________")

# Feature-wise churn rate by segment
for col in selected_columns:
    print("\n", col)
    display(data.groupby(['segment', col])['Churn1'].mean().reset_index().T)
    print("________________________")

# Cross-tab churn % by segment and binary feature = 1
features = ['Contract_TY', 'InternetService_FB', 'Partner_P']
for col in features:
    print(f"Churn rates: segment + {col}")
    display(data.groupby(['segment', col])['Churn1'].mean().unstack())
    print("________________________")

# Segment + Churn1 percentage distribution for binary features
binary_features = [col for col in feature_columns if data[col].nunique() <= 2]
churned_data = data[data['Churn1'] == 1]

for col in binary_features:
    print(f"\n▶ Feature: {col} → Percentage distribution by Segment (only Churn1 = 1)\n")
    pivot = pd.crosstab(
        index=churned_data['segment'],
        columns=churned_data[col],
        normalize='index'
    ) * 100
    pivot.columns = [f"{col} = {val}" for val in pivot.columns]
    pivot = pivot.reset_index()
    display(pivot.round(2))
    print("______________________________")

# V2: Distribution of segment membership when binary feature = 1 and Churn1 = 1
from functools import reduce
summary_list = []

for col in binary_features:
    temp = churned_data[churned_data[col] == 1]
    counts = temp['segment'].value_counts(normalize=True) * 100
    summary_list.append(counts.rename(col))

summary_df = pd.concat(summary_list, axis=1).T.fillna(0).round(2)
display(summary_df)

# Visualizing above as stacked bar chart
summary_df.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='coolwarm_r')
plt.title('% Distribution of Segment for Each Feature (where Feature = 1 and Churn1 = 1)', fontsize=13)
plt.ylabel('Percentage')
plt.xlabel('Features (binary = 1)')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Segment', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.show()
