# Navid Tarighati – Portfolio

Welcome to my data portfolio. Below are selected projects that showcase my experience in analytics and BI tools.

---

## 📊 Sales Intelligence Dashboard Suite
**Tools**: Tableau Desktop Professional, Tableau Public  

This Tableau project includes **three dashboards** focused on:
- 📈 Profit & Sales Analysis  
- 📦 Product Performance  
- 👥 Customer Insights

Each dashboard presents **key performance metrics** alongside **three visualizations**, which can be dynamically toggled using a custom navigation menu.  
The visualizations are powered by **Dynamic Zone Visibility**, allowing users to seamlessly swap between charts for deeper analysis and interactivity.

**📷 Preview**:  
[Profit & Sales Dashboard](https://public.tableau.com/shared/62HH5D5BR?:display_count=n&:origin=viz_share_link) <br>
[Customer Dashboard](https://public.tableau.com/views/Customer-Dashboard_17447706185050/CustomerOverView?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link) <br>
[Product Dashboard](https://public.tableau.com/views/Product-Dashboard/ProductDash?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

## 📊 Insurance Claim Prediction

**Tools**: python(pandas, seaborn, sklearn)  
**Description**: This project builds a logistic regression model to predict insurance claims using a cleaned and preprocessed dataset. Data preparation involved handling missing values, transforming features, addressing outliers, and encoding categorical variables. Forward feature selection identified the top predictors, and multiple models were evaluated using 5-fold cross-validation on key metrics like precision, recall, and F1-score. The best-performing model was trained on scaled features using RobustScaler. Both the trained model and the scaler were saved as pickle files for streamlined deployment and future predictions.

**🔗 Download Report**:  [Report](./reports/Report.pdf) <br>
**🔗 Download Model Training Notebook**: [Training.py](.notebooks/Training.py) <br>
**🔗 Download Scaling Pickle**: [Scaling_pickle.pkl](./notebooks/best_model_scaler.pkl) <br>
**🔗 Download Best Model Pickle**: [Best_model.pkl](./notebooks/best_model_logistic_regression.pkl) <br>
**🔗 Download Production Notebook**: [Production.py](./notebooks/Production.py)

## 📊 Call Center Dashboard

**Tools**:  Power BI, DAX  
**Description**: Built an interactive dashboard analyzing call center performance. 
Offering a dashboard for the Customer Support Operations to help understand the performance by filtering through the agents, support ticket topics, date of ticket submission.

**📷 Preview**:  
[Call Center Dashboard](./screenshots/Call_Center.png)

**🔗 Download PBIX**: [Call-Center-dashboard.pbix](./pbix/Call_Center_Dashboard.pbix)

## 📊 Revenue & Customer Insights – Movie Rental Business

**Tools**: SQL Server  <br>
**Description**: An analysis and report for actions and insights to help the business. 
Conducted a comprehensive data analysis project on a movie rental business using SQL Server, with the goal of uncovering actionable business insights.
This project focuses on querying the database to perform detailed analyses across key operational areas including Film & Category Performance, Customer Value, Inventory Usage, Store & Staff Productivity, and Geographic & Temporal Trends.

The insights were synthesized into a structured report highlighting patterns in customer behavior, category profitability, seasonal revenue trends, and RFM-based segmentation. Key metrics such as top-performing films, loyal customers, peak months, and underperforming inventory were identified to support strategic decision-making.

**🔗 Download Report**:  [Insights and actions.pdf](./reports/Insights%20and%20actions.pdf) <br>
**🔗 Download SQL**: [movie_rentals_analysis.sql](./SQL/movie_rentals_analysis.sql)

## 📊 Customer Segmentation with K-Means – Online Retail II Dataset

**Tools**: Python (Pandas, Scikit-Learn, Seaborn, Matplotlib), Jupyter Notebook 
**Description**: Performed end-to-end customer segmentation using the Online Retail II dataset to identify behavioral clusters based on Recency, Frequency, and Monetary (RFM) values.
This project involved thorough data cleaning, outlier analysis, and feature scaling before applying K-Means clustering.
Both Elbow Method and Silhouette Score were used to determine the optimal number of clusters.
Clusters were visualized using 3D scatter plots and violin plots to understand purchasing patterns.
Insights were synthesized in two reports—one focusing on Cluster Profiles and another on Outlier Analysis—each with action-based recommendations tailored to customer value and engagement.

**🔗 Download Notebook**: [Kmeans Clustering.py](./notebooks/Kmeans_Clustering.py) <br>
**🔗 Download Cluster Analysis Report**: [Cluster Analysis.pdf](./reports/Cluster_analysis.pdf) <br>
**🔗 Download Outlier Analysis Report**: [Outlier Analysis.pdf](./reports/Outlier_Analysis.pdf)

## 📊 Customer Churn Dashboard
**Tools**:  Power BI, DAX  
**Description**: Developed an interactive dashboard to analyze customer churn patterns for a telecom company. The dashboard enables dynamic filtering by Payment Method, Gender, Partner Status, and Contract Type, offering granular insights into customer behavior.

**📷 Preview**:  
[Customer Churn Dashboard](./screenshots/churn_dashboard.png)

**🔗 Download PBIX**: [Customer_ churn_dashboard.pbix](./pbix/Churn_Dashboard.pbix)
