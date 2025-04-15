#!/usr/bin/env python
# coding: utf-8

# In[1]:


# libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


# In[2]:


#display ssettings
pd.options.display.float_format = '{:20.2f}'.format

#show all columns in output
pd.set_option('display.max_columns', 999)




# ## Data Exploration

# In[3]:


df = pd.read_excel("C:/Users/tarig/PycharmProjects/End to End/data/online_retail_II.xlsx", sheet_name=0)

df.head(5)



# In[4]:


df.info()

## customer ID has nulls



# In[5]:


df.describe()


# #### min quantity is -9600 and min price is -53594 which need further exploration

# In[6]:


df.describe(include='O')


# #### number of unique stock codes and descriptions are different and the count of descriptions is fewer which should be checked

# In[9]:


df[df["Customer ID"].isna()].head(10)


# #### since we are focused on customer data, it would be better to remove null customer IDs.

# In[10]:


df[df["Quantity"]<0].head(10)


# #### the invoices that start with C are for cancelled orders so they have negative quantities

# In[11]:


df["Invoice"] = df["Invoice"].astype("str")


# #### Checking where the invoice number is not exactly 6 digits

# In[12]:


df[df["Invoice"].str.match("^\\d{6}$")==False]


# #### Checking if C in the beginning of invoice number is the only anomaly by taking out all digits and check if C is unique

# In[13]:


df["Invoice"].str.replace("[0-9]", "", regex=True).unique()


# #### We can see that in addition to cancelled orders that start with a C, there are orders that have the letter A in them

# In[14]:


df[df["Invoice"].str.startswith("A")]


# #### All these are described as "adjust bad debt" so these can be removed especially since the customer ID is NaN

# In[15]:


df["StockCode"]= df["StockCode"].astype("str")
df[df["StockCode"].str.match("^6\\d{5}$")==False]


# #### There are a very large number of Stock Codes which do not follow the pattern in the documentation which is 6 digits only

# In[16]:


df[df["StockCode"].str.match("^6\\d{5}$") & df["StockCode"].str.match("^6\\d{5}[a-zA-Z]+$")==False]


# ### Ideally, we want to check these unusual Stock Codes to find out if they are useful or not.

# In[18]:


# Checking if we have stock codes that are not unique to a description
df[df.duplicated(subset=["Description"], keep=False)]



# ## Data Cleaning

# In[19]:


cleaned_df=df.copy()


# #### Cleaning the invoices for cancellation and accounting. We don't need them since we are not working on sales but Customer Clustering

# In[20]:


cleaned_df["Invoice"]=cleaned_df["Invoice"].astype("str")
# Create a filter expression
mask = (

    cleaned_df["Invoice"].str.match("^\\d{6}$") == True
)


# In[21]:


cleaned_df = cleaned_df[mask]


# In[22]:


cleaned_df


# #### Cleaning the Stock Codes

# In[23]:


cleaned_df["StockCode"]=cleaned_df["StockCode"].astype("str")
# Create a filter expression
mask = (
    (cleaned_df["StockCode"].str.match("^\\d{5}$") == True) |
    (cleaned_df["StockCode"].str.match("^\\dd{5}[a-zA-Z]+$") == True) |
    (cleaned_df["StockCode"].str.match("^PADS$") == True) 
)


# In[24]:


cleaned_df = cleaned_df[mask]
cleaned_df


# In[25]:


# droping null customer values
cleaned_df.dropna(subset=["Customer ID"], inplace=True)


# #### we need to check that negative price is removed

# In[26]:


cleaned_df.describe()


# In[27]:


## min price and quantity no more have negative values but Price has a min of 0
cleaned_df[cleaned_df["Price"]==0]


# In[28]:


# How many of these records do we have?
len(cleaned_df[cleaned_df["Price"]==0])


# #### we can't see any reason for the 0 price but it could be result of a giveaway or a sales event.
# #### At this time, we remove them since we have no idea where they come from and they are only 27 instances

# In[29]:


cleaned_df = cleaned_df[cleaned_df["Price"]>0]
cleaned_df.describe()


# In[30]:


cleaned_df["Price"].min()


# #### seems like we have one row where the price is 0.0001 so it appears as a zero 

# In[31]:


cleaned_df[cleaned_df["Price"] ==0.001].head()


# #### We can keep these in since there are only 5 of them

# In[32]:


### Let's check how much data have we dropped during cleaning
len(cleaned_df)/len(df)


# ## We have dropped 33% of our data while cleaning

# ### We want to explore RFM

# In[33]:


# adding a total spent for each line
cleaned_df["LineTotal"]=cleaned_df["Quantity"]*cleaned_df["Price"]


# In[34]:


cleaned_df


# ### Creating RFM by aggregating our Data

# In[35]:


aggregated_df = cleaned_df.groupby(by="Customer ID", as_index=False) \
    .agg(
        MonetaryValue=("LineTotal","sum"),
        Frequency=("Invoice","nunique"),
        LastInvoiceDate=("InvoiceDate", "max")
)
aggregated_df.head(5)


# #### Cleaning the Customer ID column datatype

# In[ ]:





# In[36]:


max_invoice_date= aggregated_df["LastInvoiceDate"].max()


# ### Since data is old, I am using the latest invoice date as today for Recency

# In[37]:


aggregated_df["Recency"]= (max_invoice_date - aggregated_df["LastInvoiceDate"]).dt.days

aggregated_df.head(5)


# In[38]:


aggregated_df.describe()


# #### finding outliers

# In[39]:


plt.figure(figsize=(15,5))

plt.subplot(1, 3, 1)
plt.hist(aggregated_df['MonetaryValue'], bins=10, color='skyblue', edgecolor='black')
plt.title('Monetary Value Distribution')
plt.xlabel('Monetary Value')
plt.ylabel('count')


plt.subplot(1, 3, 2)
plt.hist(aggregated_df['Frequency'], bins=10, color='lightgreen', edgecolor='black')
plt.title('Frequency Distribution')
plt.xlabel('Frequency Value')
plt.ylabel('count')


plt.subplot(1, 3, 3)
plt.hist(aggregated_df['Recency'], bins=10, color='salmon', edgecolor='black')
plt.title('Recency Distribution')
plt.xlabel('Recency Value')
plt.ylabel('count')

plt.tight_layout()
plt.show()


# In[40]:


plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
sns.boxplot(data=aggregated_df['MonetaryValue'], color= 'skyblue')
plt.title('Monetary Value BoxPlot')
plt.xlabel("Monetary Value")

plt.subplot(1,3,2)
sns.boxplot(data=aggregated_df['Frequency'], color= 'lightgreen')
plt.title('Frequency BoxPlot')
plt.xlabel("Frequency")

plt.subplot(1,3,3)
sns.boxplot(data=aggregated_df['Recency'], color= 'salmon')
plt.title('Recency BoxPlot')
plt.xlabel("Recency")

plt.tight_layout()
plt.show()


# ### The outliers are high valued customers, so getting rid of them is a mistake when we are discussing customer segmentation
# ### The proper way to go forward is to seperate them as their own cluster

# In[41]:


aggregated_df


# In[42]:


# writing new quartile range for Monetary Value
M_Q1 = aggregated_df["MonetaryValue"].quantile(0.25)
M_Q3 = aggregated_df["MonetaryValue"].quantile(0.75)
M_IQR = M_Q3-M_Q1

monetary_outliers_df= aggregated_df[(aggregated_df["MonetaryValue"] > (M_Q3+ 1.5 * M_IQR))|(aggregated_df["MonetaryValue"] < (M_Q1 - 1.5 * M_IQR))].copy()


# In[43]:


monetary_outliers_df.describe()


# In[44]:


# writing new quartile range for Frequency
M_Q1 = aggregated_df["Frequency"].quantile(0.25)
M_Q3 = aggregated_df["Frequency"].quantile(0.75)
M_IQR = M_Q3-M_Q1

frequency_outliers_df= aggregated_df[(aggregated_df["Frequency"] > (M_Q3+ 1.5 * M_IQR))|(aggregated_df["Frequency"] < (M_Q1 - 1.5 * M_IQR))].copy()
frequency_outliers_df.describe()


# In[45]:


# writing new quartile range for Recency
M_Q1 = aggregated_df["Recency"].quantile(0.25)
M_Q3 = aggregated_df["Recency"].quantile(0.75)
M_IQR = M_Q3-M_Q1

Recency_outliers_df= aggregated_df[(aggregated_df["Recency"] > (M_Q3+ 1.5 * M_IQR))|(aggregated_df["Recency"] < (M_Q1 - 1.5 * M_IQR))].copy()
Recency_outliers_df.describe()


# ## Let's find the non-outliers

# In[46]:


#### Since we have not reset the indeces 


# In[47]:


non_outliers_df = aggregated_df[(~aggregated_df.index.isin(monetary_outliers_df.index))&(~aggregated_df.index.isin(frequency_outliers_df.index))]
non_outliers_df.describe()


# ### There is a good amount of reduction of difference between mean and Std due to trimming the outliers

# In[48]:


plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
sns.boxplot(data=non_outliers_df['MonetaryValue'], color= 'skyblue')
plt.title('Monetary Value BoxPlot')
plt.xlabel("Monetary Value")

plt.subplot(1,3,2)
sns.boxplot(data=non_outliers_df['Frequency'], color= 'lightgreen')
plt.title('Frequency BoxPlot')
plt.xlabel("Frequency")

plt.subplot(1,3,3)
sns.boxplot(data=non_outliers_df['Recency'], color= 'salmon')
plt.title('Recency BoxPlot')
plt.xlabel("Recency")

plt.tight_layout()
plt.show()


# In[49]:


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(projection="3d")
scatter = ax.scatter(non_outliers_df["MonetaryValue"],non_outliers_df["Frequency"],non_outliers_df["Recency"])

ax.set_xlabel("Monetary Value")
ax.set_ylabel("Frequency")
ax.set_zlabel("Recency")

ax.set_title('3D Scatter Plot of Customer Data')
plt.show()


# ### !Important to notice that due to the difference of scales, the data is clustered in one corner
# #### This matters since the Kmean cluster is sensitive and centroids will be shifted to monetary value

# In[50]:


# we will use z score for rescaling
scaler = StandardScaler()


# In[51]:


scaled_data = scaler.fit_transform(non_outliers_df[["MonetaryValue", "Frequency","Recency"]])
scaled_data


# In[52]:


scaled_data_df = pd.DataFrame(scaled_data, index=non_outliers_df.index, columns=("MonetaryValue","Frequency","Recency"))
scaled_data_df


# In[53]:


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(projection="3d")
scatter = ax.scatter(scaled_data_df["MonetaryValue"],scaled_data_df["Frequency"],scaled_data_df["Recency"])

ax.set_xlabel("Monetary Value")
ax.set_ylabel("Frequency")
ax.set_zlabel("Recency")

ax.set_title('3D Scatter Plot of Customer Data')
plt.show()


# ### We use Elbow methods in conjunction with silhouette to determine the value of k

# In[54]:


# assign a max number of centroids
max_k = 12

inertia = []

#define the range of possible k values
k_values = range(2, max_k+1)

for k in k_values:
    
    kmeans= KMeans(n_clusters=k, random_state=40, max_iter=1000)
    
    kmeans.fit_predict(scaled_data_df)
    
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(14,6))
plt.plot(k_values, inertia, marker="o")

plt.title('KMeans Inertia for Different Values of k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.xticks(k_values)
plt.grid(True)

plt.show()


# #### We can see that the trajectory of the graph stabilizes near 4 and 5

# ### Silhouette Score will help decide between 4 and 5

# In[55]:


max_k = 12

inertia = []
silhouette_scores=[]
#define the range of possible k values
k_values = range(2, max_k+1)

for k in k_values:
    
    kmeans= KMeans(n_clusters=k, random_state=42, max_iter=1000)
    
    cluster_labels=kmeans.fit_predict(scaled_data_df)
    
    sil_score= silhouette_score(scaled_data_df,cluster_labels)
    
    silhouette_scores.append(sil_score)
    
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(14,6))

plt.subplot(1,2,1)
plt.plot(k_values, inertia, marker="o")
plt.title('KMeans Inertia for Different Values of k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.xticks(k_values)
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(k_values, silhouette_scores, marker="o", color='green')
plt.title('Silhouette Scores for Different Values of k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette cores')
plt.xticks(k_values)
plt.grid(True)

plt.show()


# #### in the sillhouette scores graph, we can see that 4 is more stable than 5

# In[56]:


kmeans = KMeans(n_clusters=4, random_state=42, max_iter=1000)

cluster_labels = kmeans.fit_predict(scaled_data_df)

cluster_labels


# In[59]:


non_outliers_df["Cluster"]=cluster_labels
non_outliers_df


# In[60]:


cluster_colors = {
    0: '#1f77bf',
    1: '#ff7f0e',
    2: '#2ca02c',
    3: '#d62728'
}

colors=non_outliers_df['Cluster'].map(cluster_colors)

fig = plt.figure(figsize=(10,10))
ax=fig.add_subplot(projection='3d')

scatter = ax.scatter(
    non_outliers_df['MonetaryValue'],
    non_outliers_df['Frequency'],
    non_outliers_df['Recency'],
    c=colors,
    marker='o'
)

ax.set_xlabel('Monetary Value')
ax.set_ylabel('Frequency')
ax.set_zlabel('Recency')

ax.set_title('3D Scatter Plot of Customer Data by Cluster')
plt.show()


# ### We can use a violin plot to drill down on each 3 axes

# In[72]:


plt.figure(figsize=(12, 18))

plt.subplot(3, 1, 1)
sns.violinplot(x=non_outliers_df['Cluster'], y=non_outliers_df['MonetaryValue'], palette=cluster_colors, hue=non_outliers_df["Cluster"])
#sns.violinplot(y=non_outliers_df['MonetaryValue'], color='gray', linewidth=1.0)
plt.title('Monetary Value by Cluster')
plt.ylabel('Monetary Value')

plt.subplot(3, 1, 2)
sns.violinplot(x=non_outliers_df['Cluster'], y=non_outliers_df['Frequency'], palette=cluster_colors, hue=non_outliers_df["Cluster"])
#sns.violinplot(y=non_outliers_df['Frequency'], color='gray', linewidth=1.0)
plt.title('Frequency by Cluster')
plt.ylabel('Frequency')


plt.subplot(3, 1, 3)
sns.violinplot(x=non_outliers_df['Cluster'], y=non_outliers_df['Recency'], palette=cluster_colors, hue=non_outliers_df["Cluster"])
#sns.violinplot(y=non_outliers_df['Recency'], color='gray', linewidth=1.0)
plt.title('Recency by Cluster')
plt.ylabel('Recency')

plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:




