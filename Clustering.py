#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np


# In[7]:


train_df = pd.concat((pd.read_csv('/Users/joel/Downloads/N/Output/NotEncoded_Genres_ArtistsEncoded/p%ofgenre/Train.csv'), pd.read_csv('/Users/joel/Downloads/N/Output/NotEncoded_Genres_ArtistsEncoded/p%ofgenre/Test.csv')))


# In[8]:


X_train = train_df.drop(columns=['genres'])
y_train = train_df['genres']


# In[9]:


# Classification problem requires validation and test set, clustering does not
# We use a 60-20-20 split for classification, and all of the data for clustering


# In[10]:


X_train


# In[11]:


artists_encoded_train = X_train[['Artists_encoded','key','mode']]
X_train = X_train.drop(columns=['Artists_encoded','key','mode'])


# In[12]:


X_train


# In[13]:


scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)


# In[14]:


X_train_scaled.isna().sum()


# In[15]:


artists_encoded_train = artists_encoded_train.reset_index(drop=True)

X_train_scaled[['Artists_encoded','key','mode']] = artists_encoded_train

print("Training set shape:", X_train_scaled.shape, y_train.shape)


# In[16]:


X_train_scaled


# In[17]:


X_train_scaled.isna().sum()


# In[18]:


corr_matrix = X_train_scaled.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()


# In[19]:


corr_matrix = X_train_scaled.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.75)]
X_selected = X_train_scaled.drop(columns=to_drop)
X_selected


# In[20]:


from sklearn.ensemble import RandomForestClassifier


rf_model = RandomForestClassifier()
rf_model.fit(X_train_scaled, y_train)

feature_importance = rf_model.feature_importances_

feature_importance_df = pd.DataFrame({'Feature': X_train_scaled.columns, 'Importance': feature_importance})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("Top Features Selected by Random Forest:")
print(feature_importance_df)


# In[21]:


X_train_scaled1 = X_train_scaled.drop(columns=['mode','energy'])


# In[22]:


X_train_scaled1


# In[24]:


from sklearn.cluster import KMeans

# Choose the number of clusters (K)
# We are choosing 100 classifiers because we know that our labelled data truly falls into 100 unique genres
n_clusters = 100

# Fit the KMeans model on the scaled data
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_train_scaled1)

# Predict cluster labels
train_labels = kmeans.predict(X_train_scaled1)

# Analyze results
# For example, you can check the size of each cluster
print("Cluster counts in training data:")
print(pd.Series(train_labels).value_counts())

# You can also examine the cluster centers
print("Cluster centers:")
print(kmeans.cluster_centers_)

# Add cluster labels to your DataFrame
X_train_scaled1['Cluster'] = train_labels


# In[ ]:




