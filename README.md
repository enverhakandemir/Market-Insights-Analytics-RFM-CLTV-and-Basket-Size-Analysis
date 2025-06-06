# Market-Insights-Analytics-RFM-CLTV-and-Basket-Size-Analysis
This project presents an end-to-end customer analytics pipeline built on real-world retail data. Leveraging Python, PostgreSQL, and Power BI, we performed deep analytical processing and segmentation to uncover actionable insights for strategic decision-making in the e-commerce domain.

# Note: 
1- While the code examples in this repository use Excel files for demonstration purposes, all analyses were originally performed on a live PostgreSQL database connection in the actual project environment.
For examples of PostgreSQL integration and live data querying, please refer to my other GitHub repository: [SQL-Database-Connectivity](https://github.com/enverhakandemir/SQL-Database-Connectivity-and-Data-Exctraction-in-Python-Microsoft-SQL-MySQL-and-PostgreSQL).
2- This project was presented via an interactive Power BI dashboard connected live to a SQL database. A static preview of the report is available as a PDF in this repository (PowerBI-4MULA-CRM-PROJECT.pdf) for demonstration purposes.

# Key Components:
* Data Cleaning & Preprocessing: Over 2,000 lines of Python were used to clean, transform, and enrich the dataset for accurate downstream analysis.
* RFM Analysis: Segmented customers based on Recency, Frequency, and Monetary metrics to identify loyalty patterns and risk segments.
* CLTV Calculation: Estimated Customer Lifetime Value using transaction data and churn probability to assess long-term customer profitability.
* Basket Size & Market Basket Analysis: Applied Apriori algorithm to extract association rules and visualize product affinity via network graphs.
* Unsupervised Learning: Employed K-Means clustering (with Elbow, Silhouette, PCA, t-SNE) to discover natural customer segments.
* Supervised Learning: Built a Decision Tree model to predict churn and validate cluster integrity through classification insights.

# Outcomes:
* Personalized campaign strategies tailored to customer segments.
* Optimized retention tactics based on churn risk scoring.
* Product bundling opportunities identified via market basket rules.
* Dashboard visualizations built in Power BI for executive insights.

