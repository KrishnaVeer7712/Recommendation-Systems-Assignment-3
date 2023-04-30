import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from joblib import dump, load

anime_data=pd.read_csv('anime.csv')
rating_data=pd.read_csv('rating.csv')
anime_fulldata=pd.merge(anime_data,rating_data,on='anime_id',suffixes= ['', '_user'])
anime_fulldata = anime_fulldata.rename(columns={'name': 'anime_title', 'rating_user': 'user_rating'})

# save the model
with open('model_anime_dataset.pkl', 'wb') as f:
    pickle.dump(anime_fulldata, f)

# Loading model to compare the results
model = pickle.load(open('model_anime_dataset.pkl','rb'))
# print(model.recommend(key))
