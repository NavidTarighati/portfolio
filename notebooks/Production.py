#!/usr/bin/env python
# coding: utf-8

# In[7]:


#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import pickle
import os
from sklearn.preprocessing import RobustScaler  # Use RobustScaler as used in training

# Define the directory where the pickle files are stored
pickle_directory = 

# Provide directory for the mystery input file
input_file = 
# Load mystery dataset
df = pd.read_csv(input_file)

# **Fix `Engin_size` by removing 'L' and converting to float**
if 'Engin_size' in df.columns:
    df['Engin_size'] = df['Engin_size'].astype(str).str.rstrip('L').astype(float)

# **Convert categorical variables into dummy variables (Same as in Training)**
categorical_columns = ['Maker', 'Model', 'Color', 'Bodytype', 'Gearbox', 'Fuel_type', 'issue']
df = pd.get_dummies(df, columns=categorical_columns, dtype=int)  # ✅ Matches training approach

# **Define the best feature set from the trained model**
best_features = [
    'Adv_month', 'Engin_size', 'Seat_num', 'Door_num', 'repair_cost', 'repair_hours',
        'category_anomaly', 'Maker_Ford'
]

# **Ensure all features used in training exist in `df`**
for feature in best_features:
    if feature not in df.columns:
        df[feature] = 0  # **If missing, add it and fill with 0**

# **Select only the required features**
X = df[best_features]

# **Handle missing values (DO NOT DROP ROWS, Fill with Median as per training)**
X = X.fillna(X.median())

# **Load the trained model**
model_path = os.path.join(pickle_directory, "best_model_logistic_regression.pkl")
scaler_path = os.path.join(pickle_directory, "best_model_scaler.pkl")

with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

# **Load the scaler if used during training**
with open(scaler_path, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# **Apply Robust Scaling (Must match what was used in training)**
X_scaled = scaler.transform(X)

# **Make predictions**
predictions = model.predict(X_scaled)

# **Output predictions to CSV**
output_df = pd.DataFrame({"Prediction": predictions})
output_df.to_csv("predictions.csv", index=False)

print("✅ Predictions saved to predictions.csv successfully!")


# In[ ]:




