#!/usr/bin/env python
# coding: utf-8

# In[51]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[52]:


csv_file_path = "C:\\Users\\SREE NITHYA\\Downloads\\uber_rides_data.xlsx - sample_train.csv"
df = pd.read_csv(csv_file_path)
df


# In[53]:


csv_file_path = "C:\\Users\\SREE NITHYA\\Downloads\\uber_rides_data.xlsx - sample_train.csv"
df = pd.read_csv(csv_file_path)
df.head()


# In[54]:


df.columns


# In[55]:


df.info()


# In[56]:


# Check for missing values in the entire DataFrame
missing_values = df.isnull().sum()


# In[57]:


# Print the count of missing values for each column
print("dropoff_longitude")
print(missing_values)


# In[58]:


import pandas as pd

# Assuming 'df' is your DataFrame
df.dropna(inplace=True)
df.head()


# In[59]:


# Calculate the average fare_amount
average_fare = df['fare_amount'].mean()

# Print the average fare amount
print("Average Fare Amount:", average_fare)


# In[ ]:





# In[60]:


sns.pairplot(df, hue='passenger_count')


# In[61]:


y = df['fare_amount']
X = df[['ride_id','passenger_count','pickup_datetime','pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude']]


# In[ ]:





# In[62]:


y.head()


# In[63]:


X.head()


# In[64]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[67]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_transformed = scaler.fit_transform(X_train)
print(X_train_transformed.shape)


# In[66]:


import pandas as pd

# Sample DataFrame with a datetime column 'pickup_datetime'
data = {'pickup_datetime': ['2011-08-10 01:29:00 UTC', '2011-08-11 02:30:00 UTC', '2011-08-12 03:31:00 UTC']}
df = pd.DataFrame(data)

# Convert 'pickup_datetime' column to datetime data type
df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

# Now, you can work with datetime values in the 'pickup_datetime' column
print(df)


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_transformed = scaler.fit_transform(X_train)
print(X_train_transformed.shape)


# In[ ]:


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
X_train_transformed = scaler.fit_transform(X_train)
print(X_train_transformed.shape)


# In[ ]:


#Training


# In[ ]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train_transformed, y_train)


# In[ ]:


#Prediction
y_test_pred = classifier.predict(X_test_transformed)


# In[ ]:


#Training - KNN Classifier
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()
classifier.fit(X_train_transformed, y_train)

y_test_pred = classifier.predict(X_test_transformed)

metrics.accuracy_score(y_test, y_test_pred)


# In[ ]:


#Training - DT Classifier
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train_transformed, y_train)

y_test_pred = classifier.predict(X_test_transformed)

metrics.accuracy_score(y_test, y_test_pred)


# In[ ]:


#Training - Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train_transformed, y_train)

y_test_pred = classifier.predict(X_test_transformed)

metrics.accuracy_score(y_test, y_test_pred)


# In[ ]:





# In[ ]:




