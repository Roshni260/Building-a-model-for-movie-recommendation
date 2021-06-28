#!/usr/bin/env python
# coding: utf-8

# ## Here I have build a model for movie recommendation system.
# #### This model will recommend you movies on the basis of your choice. I have used KNN Algorithm for building this model.😊

# In[1]:


import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


movies= pd.read_csv('movies.csv')
print(movies.head())


# In[4]:


ratings= pd.read_csv('ratings.csv')
print(ratings.head())


# In[5]:


print(movies.info())
print(ratings.info())


# In[6]:


final_dataset = ratings.pivot(index='movieId',columns='userId',values='rating')
print(final_dataset.head())


# In[10]:


final_dataset.fillna(0,inplace=True)
print(final_dataset.head())


# In[17]:


no_user_voted = ratings.groupby('movieId')['rating'].agg('count')


# In[21]:


plt.scatter(no_user_voted.index,no_user_voted,color='maroon')
plt.xlabel('MovieId')
plt.ylabel('No. of users voted')
plt.show()


# ### Building KNN model for recommendation system

# In[22]:


csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)


# In[23]:


knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)


# In[24]:


def get_recommendation(movie_name):
    no_movies_to_recommend = 5  #5 movies will be recommended.You can change it according to your convinience 
    movie_list =movies[movies['title'].str.contains(movie_name)]
    if len(movie_name):
        movie_idx= movie_list.iloc[0]['movieId']
        movie_idx = final_dataset[final_dataset['movieId'] == movie_idx].index[0]
        distances , indices = knn.kneighbors(csr_data[movie_idx],n_neighbors=no_movies_to_recommend+1)    
        recommend_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
        recommend_frame = []
        for val in recommend_movie_indices:
            movie_idx = final_dataset.iloc[val[0]]['movieId']
            idx = movies[movies['movieId'] == movie_idx].index
            recommend_frame.append({'Title':movies.iloc[idx]['title'].values[0],'Distance':val[1]})
        df = pd.DataFrame(recommend_frame,index=range(1,no_movies_to_recommend+1))
        return df
    else:
        return "Oops☹!!No movies found. Please check your input"


# ### Hello Friends😊!! Now you can get recommendation of movies of your choice.

# In[27]:


get_recommendation('Avengers')


# In[28]:


get_recommendation('Toy Story')


# In[32]:


get_recommendation('Iron Man')

