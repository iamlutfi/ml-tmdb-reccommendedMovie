import streamlit as st
import pickle
import pandas as pd
from difflib import get_close_matches
import requests
from dotenv import load_dotenv
import os

# get API key
load_dotenv()
TMDB_API_KEY = os.getenv('TMDB_API_KEY') 

# get absolute path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
MODEL_DIR = os.path.join(ROOT_DIR, 'model', 'cosine_sim.pkl')
MOVIES_CSV_PATH = os.path.join(ROOT_DIR, 'data', 'preprocessed', 'pre_movies.csv')
MOVIES_LIST_PATH = os.path.join(ROOT_DIR, 'data', 'preprocessed', 'movies_list.pkl')

# --- Load Data ---
@st.cache_resource
def load_data():
    with open(MODEL_DIR, 'rb') as f:
        cosine_sim = pickle.load(f)
    movies = pd.read_csv(MOVIES_CSV_PATH)
    movies_list = pd.read_pickle(MOVIES_LIST_PATH)
    return cosine_sim, movies, movies_list

cosine_sim, movies, movies_list = load_data() 

# --- Recommendations Function ---
indices = pd.Series(movies.index, index=movies['title'])
indices = indices[~indices.index.duplicated(keep='last')]

def get_recommendations(title, cosine_sim=cosine_sim):
    close_matches = get_close_matches(title, movies['title'], n=1, cutoff=0.6)
    if not close_matches:
        return 'Movie not found in the dataset.'
    
    matched_title = close_matches[0]
    print(f'Show Recommendations for: {matched_title}')
    
    idx = indices[matched_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x:x[1], reverse=True)
    sim_scores = sim_scores[1:21]
    movie_indices = [i[0] for i in sim_scores]
    
    candidates = movies.iloc[movie_indices][['movie_id', 'title', 'vote_count', 'vote_average', 'score']].copy()
    candidates['similarity'] = [i[1] for i in sim_scores]
    candidates['final_score'] = (candidates['similarity'] * 0.7) + (candidates['score'] * 0.3)
    
    return candidates.sort_values('final_score', ascending=False).head(10)

# --- Get Poster path by API ---
def get_poster(movie_id):
    response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}')
    data = response.json()
    print(data)
    try:
        return f'https://image.tmdb.org/t/p/w500{data['poster_path']}'
    except:
        return None

# --- streamlit app ---
st.title('Movie Recommendation System')

selected_movieName = st.selectbox('Select a movie:', movies_list)

if st.button('Recommend'):
    results = get_recommendations(selected_movieName)
    if isinstance(results, str):
        st.write(results)
    else:
        st.subheader(f'Recommendations for: {selected_movieName} ')
        cols = st.columns(5)
        for i, (_, row) in enumerate(results.iterrows()):
            with cols[i % 5]:
                poster_url = get_poster(row['movie_id'])
                if poster_url:
                    st.image(poster_url)
                st.caption(f'{row['title']}')
                st.caption(f'Score: {row['score']:.2f}')