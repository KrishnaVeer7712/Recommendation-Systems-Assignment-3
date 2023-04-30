import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import os
from scipy.sparse.linalg import svds


app = Flask(__name__)
templates_dir = os.path.join(app.root_path, 'templates')
model = pickle.load(open('model_Item_user_pivot_table.pkl', 'rb'))
g_rec_list=['.hack//Liminality', '.hack//Roots', '.hack//Liminality', '.hack//Gift', '.hack//G.U. Trilogy', '.hack//Unison', '.hack//Intermezzo', '.hack//G.U. Returner', '.hack//Tasogare no Udewa Densetsu: Offline de Aimashou', '.hack//Quantum']

@app.route('/')
def home():
    # load the model
    with open('model_anime_dataset.pkl', 'rb') as f:
        model = pickle.load(f)
    # if request.method == 'POST':
    #     key = request.values.get('top anime') 
    # get recommendations based on user input
    # key = request.args.get('key')
    # if key is None:
    #     return jsonify({'error': 'Please provide a "key" parameter in the URL.'})
    # key = int(key)
    # top_anime_list = model.sort_values(["members"],ascending=False)["name"][:key].tolist()
    # top_anime_ratings = model.sort_values(["members"],ascending=False)["rating"][:key].tolist()
    # anime_data=pd.read_csv('anime.csv')
    # rating_data=pd.read_csv('rating.csv')
    # anime_fulldata=pd.merge(anime_data,rating_data,on='anime_id',suffixes= ['', '_user'])
    # anime_fulldata = anime_fulldata.rename(columns={'name': 'anime_title', 'rating_user': 'user_rating'})

    anime_fulldata= model
    # anime_fulldata = anime_fulldata.rename(columns={'name': 'anime_title', 'rating_user': 'user_rating'})
    
    #Top 10 Anime based on rating counts
    combine_anime_rating = anime_fulldata.dropna(axis = 0, subset = ['anime_title'])
    anime_ratingCount = (combine_anime_rating.groupby(by = ['anime_title'])['user_rating'].count().reset_index().rename(columns = {'rating': 'totalRatingCount'})[['anime_title', 'user_rating']])
    
    top10_animerating = anime_ratingCount[['anime_title', 'user_rating']].sort_values(by = 'user_rating',ascending = False).head(10)
    top10_animerating_user_title= top10_animerating['anime_title']
    top_anime=""
    for i in top10_animerating_user_title:
        top_anime=top_anime+ i +"  ,  "

    
    anime_fulldata = anime_fulldata.merge(anime_ratingCount, left_on = 'anime_title', right_on = 'anime_title', how = 'left')
    anime_fulldata = anime_fulldata.rename(columns={'user_rating_x': 'user_rating', 'user_rating_y': 'totalratingcount'})

    # Top 10 Anime based on Community size
    duplicate_anime=anime_fulldata.copy()
    duplicate_anime.drop_duplicates(subset ="anime_title", keep = 'first', inplace = True)
    top10_animemembers=duplicate_anime[['anime_title', 'members']].sort_values(by = 'members',ascending = False).head(10)
    top_anime_mem=""
    for i, j in zip(top10_animemembers['anime_title'], top10_animemembers['members']):
        top_anime_mem=top_anime_mem+ i+ " (Count: "+str(j) +") , "

    
    return render_template('index.html', prediction_text1=top_anime, prediction_text2=top_anime_mem)

@app.route('/content_based')
def content_based():
    return render_template('content_based.html')

@app.route('/content_based1/<int:id>')
def content_based1(id):
    global g_rec_list
    # load the model
    with open('model_Item_user_pivot_table.pkl', 'rb') as f:
        model = pickle.load(f)
    anime_pivot= model
    anime_matrix = csr_matrix(anime_pivot.values)
    model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
    model_knn.fit(anime_matrix)
    clicked_item=g_rec_list[id]
    df=anime_pivot.reset_index(drop=False)
    vect=df[df['anime_title'] == clicked_item].iloc[0, 1:].values
    query_index=np.random.choice(anime_pivot.shape[0])
    # print(query_index)
    distances, indices = model_knn.kneighbors(vect.reshape(1, -1), n_neighbors = 11)
    rec_list=[]
    for i in range(1, len(distances.flatten())):
        rec_list.append(anime_pivot.index[indices.flatten()[i]])
        #if i == 0:
        #    return render_template('index.html', prediction_text='Recommendations for {0}:\n'.format(anime_pivot.index[query_index])) 
        #else:
        #    return render_template('index.html', prediction_text='{0}: {1}, with distance of {2}:'.format(i, anime_pivot.index[indices.flatten()[i]], distances.flatten()[i])) 
    g_rec_list=rec_list
    return render_template('content_based1.html', i0=rec_list[0], i1=rec_list[1], i2=rec_list[2], i3=rec_list[3], i4=rec_list[4], i5=rec_list[5], i6=rec_list[6], i7=rec_list[7], i8=rec_list[8], i9=rec_list[9])

@app.route('/collaborative_knn')
def collaborative_knn():
    # global g_rec_list
    # load the model
    with open('model_Item_user_pivot_table.pkl', 'rb') as f:
        model = pickle.load(f)
    
    anime_pivot= model
    anime_matrix = csr_matrix(anime_pivot.values)
    model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
    model_knn.fit(anime_matrix)
    #query_index = np.random.choice(anime_pivot.shape[0])
    query_index=150
    # print(query_index)
    distances, indices = model_knn.kneighbors(anime_pivot.iloc[query_index,:].values.reshape(1, -1), n_neighbors = 11)
    collaborative_list=[]
    for i in range(1, len(distances.flatten())):
        my_str=""
        recomm=anime_pivot.index[indices.flatten()[i]]
        dist=distances.flatten()[i]
        my_str= str(recomm) + ", with distance of " + str(dist)
        collaborative_list.append(my_str)
        
    # g_rec_list=rec_list
    return render_template('collaborative_knn.html', prediction_text1=collaborative_list[0], prediction_text2=collaborative_list[1], prediction_text3= collaborative_list[2], prediction_text4=collaborative_list[3], prediction_text5=collaborative_list[4], prediction_text6=collaborative_list[5], prediction_text7=collaborative_list[6], prediction_text8=collaborative_list[7], prediction_text9=collaborative_list[8], prediction_text10=collaborative_list[9])

@app.route('/collaborative_cosin')
def collaborative_cosin():
    return render_template('collaborative_cosin.html')

@app.route('/svd')
def svd():
    # global g_rec_list
    # load the model
    with open('model_Item_user_pivot_table.pkl', 'rb') as f:
        model = pickle.load(f)
    
    anime_pivot= model
    animeT=anime_pivot.T
    matrix = animeT.values

    # Convert the data type of the rating matrix to float
    matrix = matrix.astype(float)

    # Apply SVD on the rating matrix to get the latent factors
    U, sigma, Vt = svds(matrix, k=10)
    sigma = np.diag(sigma)

    # Predict the ratings for each user-item pair using the dot product of U, sigma, and Vt
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt)

    # Define a function to recommend items for a given user
    def recommend_items(user_id, predicted_ratings, num_recommendations=10):
        # Get the user's predicted ratings
        user_ratings = predicted_ratings[user_id]
        print(user_ratings.shape)
        # Sort the ratings in descending order and return the top n items
        sorted_ratings = user_ratings.argsort()[::-1]
        top_n_items = sorted_ratings[:num_recommendations]
        return top_n_items
    user_id = 4
    recommendations = recommend_items(user_id, all_user_predicted_ratings, num_recommendations=10)
    svd_list=[]
    for i in recommendations:
        svd_list.append(animeT.columns[i])

    # g_rec_list=rec_list
    return render_template('svd.html', prediction_text1=svd_list[0], prediction_text2=svd_list[1], prediction_text3= svd_list[2], prediction_text4=svd_list[3], prediction_text5=svd_list[4], prediction_text6=svd_list[5], prediction_text7=svd_list[6], prediction_text8=svd_list[7], prediction_text9=svd_list[8], prediction_text10=svd_list[9])

@app.route('/recommend',methods=['GET','POST'])
def recommend():
    # load the model
    with open('model_Item_user_pivot_table.pkl', 'rb') as f:
        model = pickle.load(f)
    if request.method == 'POST':
        key = request.values.get('top anime') 
    # get recommendations based on user input
    # key = request.args.get('key')
    if key is None:
        return jsonify({'error': 'Please provide a "key" parameter in the URL.'})
    key = int(key)
    # top_anime_list = model.sort_values(["members"],ascending=False)["name"][:key].tolixst()
    # top_anime_ratings = model.sort_values(["members"],ascending=False)["rating"][:key].tolist()

    anime_pivot= model
    anime_matrix = csr_matrix(anime_pivot.values)
    model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
    model_knn.fit(anime_matrix)
    #query_index = np.random.choice(anime_pivot.shape[0])
    query_index=15
    # print(query_index)
    distances, indices = model_knn.kneighbors(anime_pivot.iloc[query_index,:].values.reshape(1, -1), n_neighbors = key+1)
    rec_list=[]
    for i in range(1, len(distances.flatten())):
        rec_list.append(anime_pivot.index[indices.flatten()[i]])
        #if i == 0:
        #    return render_template('index.html', prediction_text='Recommendations for {0}:\n'.format(anime_pivot.index[query_index])) 
        #else:
        #    return render_template('index.html', prediction_text='{0}: {1}, with distance of {2}:'.format(i, anime_pivot.index[indices.flatten()[i]], distances.flatten()[i])) 
    g_rec_list=rec_list
    return render_template('index.html', prediction_text1=rec_list[0], prediction_text2=rec_list[1])
    # top_anime_temp1 = model.sort_values(["members"],ascending=False)

    # return jsonify({'top_anime_list': top_anime_list})


if __name__ == "__main__":
    app.run(debug=True)
