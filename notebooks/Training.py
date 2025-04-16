#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[81]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import metrics
from sklearn.model_selection import train_test_split,cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.metrics import mean_squared_error, f1_score, recall_score, precision_score

df = pd.read_csv('C:/Users/tarig/OneDrive/Documents/BCIT/Winter 2025/Advance Topics in Data Analytics/Assignment 1/asgn1_vehicleClaimData/asgn1_vehicleClaimData/VehicleInsuranceClaims.csv')


# ## EDA

# In[82]:


df.shape


# In[86]:


df.describe()


# #### Describe is a valuable asset in detecting possible outliers or eskewedness of data

# ##### Runned_Miles has negative values which is clearly an issue

# In[41]:


df.info()


# In[42]:


# checking for outliers based on cues in the describe command
sns.boxplot(data=df['repair_hours'])


# In[ ]:


sns.boxplot(data=df['repair_hours'])


# In[103]:


import numpy as np

Q1 = df['repair_hours'].quantile(0.25)
Q3 = df['repair_hours'].quantile(0.75)
IQR = Q3 - Q1

# Define upper bound (1.5*IQR rule)
upper_bound = Q3 + 1.5 * IQR

# Replace outliers with mean
df['repair_hours_noOutliers'] = df['repair_hours'].apply(lambda x: df['repair_hours'].mean() if x > upper_bound else x)



# In[78]:


df['repair_hours_noOutliers'].isnull().sum()


# ##### Finding the total number of nulls in each column

# In[113]:


df.isnull().sum()


# ### Data cleaning

# #### Replacing the missing values in Claim and category anomaly with median of the column

# In[114]:


df['Claim'] = df['Claim'].fillna(df['Claim'].mode()[0])
df['category_anomaly'] = df['category_anomaly'].fillna(df['category_anomaly'].mode()[0])


# In[115]:


# checking to make sure the nulls are removed
print(df['Claim'].isna().sum())
print(df['category_anomaly'].isna().sum())


# #### Backfilling the missing values in repair_date column

# In[91]:


# checking to make sure the null is taken care of
df["breakdown_date"] = pd.to_datetime(df["breakdown_date"], errors='coerce')
df["repair_date"] = pd.to_datetime(df["repair_date"], errors='coerce')
df['repair_date']=df['repair_date'].fillna(df['repair_date'].median())
df['repair_date'].isna().sum()


# In[84]:


# Handling the negative values in Runned_Miles by replacing them with absolute values.
df["Runned_Miles"] = df["Runned_Miles"].abs()


# In[87]:


sns.boxplot(data=df['Runned_Miles'])


# ### the boxplot suggests a large number of outliers since data is compressed around the lower values.
# ### we can check further to prove this with a histogram

# In[88]:


sns.histplot(df['Runned_Miles'], bins=50, kde=True)
plt.show()


# In[94]:


#Create a new column which has outliers removed
df['Runned_Miles_noOutliers'] = df['Runned_Miles'].apply(lambda x: df['repair_hours'].mean() if x > 74000.0 else x)
df['Runned_Miles_noOutliers'] = df['Runned_Miles_noOutliers'].astype(float)


# In[96]:


# Box plot showing the runned_miles without outliers
sns.boxplot(data=df['Runned_Miles_noOutliers'])


# In[116]:


# a final check to find any missing values left
for column in df.columns:
    print(f"{column}: {df[column].isna().sum()} missing values")


# In[ ]:





# In[117]:


df['Engin_size'].head()


# In[118]:


# Engin_size is a string with L in the end so we covert it to float
df['Engin_size'] = df['Engin_size'].str.rstrip('L').astype(float)


# In[119]:


len(df.columns)


# In[120]:


# dropping the date columns and issue_id 
df.drop(columns=['repair_date', 'breakdown_date', 'issue_id'], inplace=True)


# In[121]:


# Convert categorical variables into dummy variables 
categorical_columns = ['Maker', 'Model', 'Color', 'Bodytype', 'Gearbox', 'Fuel_type', 'issue']
df = pd.get_dummies(df, columns=categorical_columns, dtype=int)


# In[122]:


### exporting the cleaned data to a csv 

df.to_csv("cleanedup_vehicle_insurance_data.csv", index=False)


# ## Data is cleaned. We will use the cleaned dataset for feature selection

# In[70]:


from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_regression

import pandas as pd


# In[71]:


df=pd.read_csv('cleanedup_vehicle_insurance_data.csv')


# In[130]:


# Seperate the target and independent variables
X = df.copy()       # Create separate copy to prevent unwanted tampering of data.
del X['Claim']     # Delete target variable.

# Target variable
y = df['Claim']



# In[131]:


# perform Forward Feature Selection, choosing 15 top features
ffs_selector = SelectKBest(score_func=f_regression, k=15)  

# fit the selector to the data
ffs_selector.fit(X, y)

# extract top 15 selected feature names
top_15_features = X.columns[ffs_selector.get_support()].tolist()

# display the top 10 features
print("\nTop 15 Features Based on FFS (f_regression):\n")
top_15_features


# ##### building the model with 15 features

# In[22]:


top_features = ['Adv_month', 'Engin_size', 'Seat_num', 'Door_num', 'repair_cost', 'repair_hours', 'category_anomaly', 'Maker_Ford', 'Model_B-Max', 'Model_Focus', 'Color_Black', 'Color_Blue', 'Color_Gelb', 'Color_Silver', 'Bodytype_Wood']


# In[23]:


from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score, make_scorer
import pandas as pd
import numpy as np

# Define feature matrix and target variable
X = df[['Adv_month', 'Engin_size', 'Seat_num', 'Door_num', 'repair_cost', 'repair_hours',
        'category_anomaly', 'Maker_Ford', 'Model_B-Max',
        'Model_Focus', 'Color_Black', 'Color_Blue', 'Color_Gelb', 'Color_Silver', 'Bodytype_Wood']]

# Use selected features
y = df['Claim']

# Store model results
first_model_results = []

# Define K-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Split into training and testing (80% train, 20% test) WITHOUT SCALING
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression
model = LogisticRegression(max_iter=1000, solver='liblinear')

# Perform Cross-Validation
accuracy_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
precision_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring=make_scorer(precision_score))
recall_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring=make_scorer(recall_score))
f1_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring=make_scorer(f1_score))

# Store results with mean & standard deviation
first_model_results.append({
    "Features": ", ".join(X.columns),  # Convert list to readable string
    "Avg Accuracy": np.mean(accuracy_scores),
    "Std Accuracy": np.std(accuracy_scores),
    "Avg Precision": np.mean(precision_scores),
    "Std Precision": np.std(precision_scores),
    "Avg Recall": np.mean(recall_scores),
    "Std Recall": np.std(recall_scores),
    "Avg F1-score": np.mean(f1_scores),
    "Std F1-score": np.std(f1_scores),
})

# Convert results to a DataFrame
results_df1 = pd.DataFrame(first_model_results)


# In[24]:


results_df1


# ### Since repair_hours had outliers, I will rebuild the model without scaling but replace the repair_hours with the new column that does not contain outliers

# In[133]:


from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score, make_scorer
import pandas as pd
import numpy as np

# Define feature matrix and target variable
X = df[['Adv_month', 'Engin_size', 'Seat_num', 'Door_num', 'repair_cost', 'repair_hours_noOutliers',
        'category_anomaly', 'Maker_Ford', 'Model_B-Max',
        'Model_Focus', 'Color_Black', 'Color_Blue', 'Color_Gelb', 'Color_Silver', 'Bodytype_Wood']]

# Use selected features
y = df['Claim']

# Store model results
first_model_results = []

# Define K-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Split into training and testing (80% train, 20% test) WITHOUT SCALING
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression
model = LogisticRegression(max_iter=1000, solver='liblinear')

# Perform Cross-Validation
accuracy_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
precision_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring=make_scorer(precision_score))
recall_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring=make_scorer(recall_score))
f1_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring=make_scorer(f1_score))

# Store results with mean & standard deviation
first_model_results.append({
    "Features": ", ".join(X.columns),  # Convert list to readable string
    "Avg Accuracy": np.mean(accuracy_scores),
    "Std Accuracy": np.std(accuracy_scores),
    "Avg Precision": np.mean(precision_scores),
    "Std Precision": np.std(precision_scores),
    "Avg Recall": np.mean(recall_scores),
    "Std Recall": np.std(recall_scores),
    "Avg F1-score": np.mean(f1_scores),
    "Std F1-score": np.std(f1_scores),
})

# Convert results to a DataFrame
results_df1_noOutliers = pd.DataFrame(first_model_results)


# In[134]:


results_df1_noOutliers


# ## We can see that the F1 has significantly dropped. This could mean either that the outliers are legitimately informing so I prefer not to use the noOutlier column

# ##### model with 12 features

# In[25]:


from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score, make_scorer
import pandas as pd
import numpy as np

# Define feature matrix and target variable
X = df[['Adv_month', 'Engin_size', 'Seat_num', 'Door_num', 'repair_cost', 'repair_hours',
        'category_anomaly', 'Maker_Ford', 'Model_B-Max',
        'Model_Focus', 'Color_Black', 'Color_Blue']]

# Use selected features
y = df['Claim']

# Store model results
first_model_results = []

# Define K-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Split into training and testing (80% train, 20% test) WITHOUT SCALING
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression
model = LogisticRegression(max_iter=1000, solver='liblinear')

# Perform Cross-Validation
accuracy_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
precision_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring=make_scorer(precision_score))
recall_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring=make_scorer(recall_score))
f1_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring=make_scorer(f1_score))

# Store results with mean & standard deviation
first_model_results.append({
    "Features": ", ".join(X.columns),  # Convert list to readable string
    "Avg Accuracy": np.mean(accuracy_scores),
    "Std Accuracy": np.std(accuracy_scores),
    "Avg Precision": np.mean(precision_scores),
    "Std Precision": np.std(precision_scores),
    "Avg Recall": np.mean(recall_scores),
    "Std Recall": np.std(recall_scores),
    "Avg F1-score": np.mean(f1_scores),
    "Std F1-score": np.std(f1_scores),
})

# Convert results to a DataFrame
results_df2 = pd.DataFrame(first_model_results)


# In[26]:


results_df2


# ##### Model with top 10 features

# In[27]:


from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score, make_scorer
import pandas as pd
import numpy as np

# Define feature matrix and target variable
X = df[['Adv_month', 'Engin_size', 'Seat_num', 'Door_num', 'repair_cost', 'repair_hours',
        'category_anomaly', 'Maker_Ford', 'Model_B-Max',
        'Model_Focus']]

# Use selected features
y = df['Claim']

# Store model results
first_model_results = []

# Define K-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Split into training and testing (80% train, 20% test) WITHOUT SCALING
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression
model = LogisticRegression(max_iter=1000, solver='liblinear')

# Perform Cross-Validation
accuracy_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
precision_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring=make_scorer(precision_score))
recall_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring=make_scorer(recall_score))
f1_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring=make_scorer(f1_score))

# Store results with mean & standard deviation
first_model_results.append({
    "Features": ", ".join(X.columns),  # Convert list to readable string
    "Avg Accuracy": np.mean(accuracy_scores),
    "Std Accuracy": np.std(accuracy_scores),
    "Avg Precision": np.mean(precision_scores),
    "Std Precision": np.std(precision_scores),
    "Avg Recall": np.mean(recall_scores),
    "Std Recall": np.std(recall_scores),
    "Avg F1-score": np.mean(f1_scores),
    "Std F1-score": np.std(f1_scores),
})

# Convert results to a DataFrame
results_df3 = pd.DataFrame(first_model_results)


# In[28]:


results_df3


# ##### Model with top 8 features

# In[29]:


from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score, make_scorer
import pandas as pd
import numpy as np

# Define feature matrix and target variable
X = df[['Adv_month', 'Engin_size', 'Seat_num', 'Door_num', 'repair_cost', 'repair_hours',
        'category_anomaly', 'Maker_Ford']]

# Use selected features
y = df['Claim']

# Store model results
first_model_results = []

# Define K-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Split into training and testing (80% train, 20% test) WITHOUT SCALING
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train logistic regression
model = LogisticRegression(max_iter=1000, solver='liblinear')

# Perform Cross-Validation
accuracy_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
precision_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring=make_scorer(precision_score))
recall_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring=make_scorer(recall_score))
f1_scores = cross_val_score(model, X_train, y_train, cv=kfold, scoring=make_scorer(f1_score))

# Store results with mean & standard deviation
first_model_results.append({
    "Features": ", ".join(X.columns),  # Convert list to readable string
    "Avg Accuracy": np.mean(accuracy_scores),
    "Std Accuracy": np.std(accuracy_scores),
    "Avg Precision": np.mean(precision_scores),
    "Std Precision": np.std(precision_scores),
    "Avg Recall": np.mean(recall_scores),
    "Std Recall": np.std(recall_scores),
    "Avg F1-score": np.mean(f1_scores),
    "Std F1-score": np.std(f1_scores),
})

# Convert results to a DataFrame
results_df4 = pd.DataFrame(first_model_results)


# In[30]:


results_df4


# ##### Trying the model with robust scaler

# In[31]:


from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, recall_score, precision_score, make_scorer
import pandas as pd
import numpy as np

# Define feature matrix and target variable
X = df[['Adv_month', 'Engin_size', 'Seat_num', 'Door_num', 'repair_cost', 'repair_hours',
        'category_anomaly', 'Maker_Ford']]

# Target variable
y = df['Claim']

# Store model results
first_model_results = []

# Define K-Fold Cross-Validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Split into training and testing (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# **Apply Feature Scaling Using RobustScaler**
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform training data
X_test_scaled = scaler.transform(X_test)  # Transform test data (using the same scaler)

# Train logistic regression
model = LogisticRegression(max_iter=1000, solver='liblinear')

# Perform Cross-Validation (using scaled features)
accuracy_scores = cross_val_score(model, X_train_scaled, y_train, cv=kfold, scoring='accuracy')
precision_scores = cross_val_score(model, X_train_scaled, y_train, cv=kfold, scoring=make_scorer(precision_score))
recall_scores = cross_val_score(model, X_train_scaled, y_train, cv=kfold, scoring=make_scorer(recall_score))
f1_scores = cross_val_score(model, X_train_scaled, y_train, cv=kfold, scoring=make_scorer(f1_score))

# Store results with mean & standard deviation
first_model_results.append({
    "Features": ", ".join(X.columns),  # Convert list to readable string
    "Avg Accuracy": np.mean(accuracy_scores),
    "Std Accuracy": np.std(accuracy_scores),
    "Avg Precision": np.mean(precision_scores),
    "Std Precision": np.std(precision_scores),
    "Avg Recall": np.mean(recall_scores),
    "Std Recall": np.std(recall_scores),
    "Avg F1-score": np.mean(f1_scores),
    "Std F1-score": np.std(f1_scores),
})

# Convert results to a DataFrame
results_df4_RbostScaled = pd.DataFrame(first_model_results)


# In[32]:


results_df4_RbostScaled


# In[33]:


# List of all DataFrames you want to merge
all_results = [results_df1, results_df2, results_df3, results_df4, results_df4_RbostScaled]  # Add all your DataFrames here

# Merge them into a single DataFrame
final_results_df = pd.concat(all_results, ignore_index=True)

# Display the merged DataFrame
final_results_df


# In[34]:


final_results_df.to_excel("C:/Users/tarig/OneDrive/Documents/BCIT/Winter 2025/Advance Topics in Data Analytics/Assignment 1/Pickles/final_results.xlsx", index=False)


# ## Saving the Scaler and Best model as pickles

# In[36]:


import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# Define feature matrix and target variable
X = df[['Adv_month', 'Engin_size', 'Seat_num', 'Door_num', 'repair_cost', 'repair_hours',
        'category_anomaly', 'Maker_Ford']]

y = df['Claim']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# **Apply RobustScaler**
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000, solver='liblinear')
model.fit(X_train_scaled, y_train)

# **Define directory to save pickle files**
pickle_directory =   "C:/Users/tarig/OneDrive/Documents/BCIT/Winter 2025/Advance Topics in Data Analytics/Assignment 1/Pickles"

# **Ensure directory exists**
os.makedirs(pickle_directory, exist_ok=True)

# **Save the trained model**
model_path = os.path.join(pickle_directory, "best_model_logistic_regression.pkl")
with open(model_path, "wb") as model_file:
    pickle.dump(model, model_file)

# **Save the RobustScaler**
scaler_path = os.path.join(pickle_directory, "best_model_scaler.pkl")
with open(scaler_path, "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print(f"✅ Model saved to: {model_path}")
print(f"✅ Scaler saved to: {scaler_path}")


# In[ ]:




