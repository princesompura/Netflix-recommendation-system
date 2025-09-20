import gradio as gr
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler
import os

# Cache implementation
_cached_movies = None
_cached_ratings = None

def load_data():
    global _cached_movies
    if _cached_movies is not None:
        return _cached_movies
    try:
        credits = pd.read_csv("tmdb_5000_credits.csv")
        movies = pd.read_csv("tmdb_5000_movies.csv")
        credits.rename(columns={'movie_id': 'id'}, inplace=True)
        credits['id'] = pd.to_numeric(credits['id'], errors='coerce')
        credits = credits[credits['id'].apply(lambda x: isinstance(x, (int, float)) and not np.isnan(x))]
        credits['id'] = credits['id'].astype(int)
        movies['id'] = movies['id'].astype(int)
        movies = movies.merge(credits, on='id', suffixes=('', '_credit'))
        if 'title_credit' in movies.columns:
            movies.drop(columns=['title_credit'], inplace=True)
        movies.drop_duplicates(subset='title', inplace=True)
        _cached_movies = movies
        return movies
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def load_ratings_data():
    global _cached_ratings
    if _cached_ratings is not None:
        return _cached_ratings
    try:
        ratings = pd.read_csv("ratings_small.csv")
        _cached_ratings = ratings
        return ratings
    except Exception as e:
        print(f"Error loading ratings data: {e}")
        return None

def predict_rating(user_id, movie_id):
    try:
        ratings = load_ratings_data()
        if ratings is None:
            return "Error: Could not load ratings data"
        
        # Create user-movie matrix
        ratings_matrix = ratings.pivot(index='userId', 
                                     columns='movieId', 
                                     values='rating').fillna(0)
        
        # Check if user_id and movie_id exist in the matrix
        if user_id not in ratings_matrix.index or movie_id not in ratings_matrix.columns:
            return "Error: User ID or Movie ID not found in dataset"
        
        # Instead of StandardScaler, we'll normalize the data to [0,1] range
        ratings_array = ratings_matrix.values
        ratings_normalized = (ratings_array - ratings_array.min()) / (ratings_array.max() - ratings_array.min())
        
        # Apply NMF
        nmf = NMF(n_components=50, init='random', random_state=42)
        user_features = nmf.fit_transform(ratings_normalized)
        movie_features = nmf.components_
        
        # Get user and movie indices
        user_idx = ratings_matrix.index.get_loc(user_id)
        movie_idx = ratings_matrix.columns.get_loc(movie_id)
        
        # Predict rating
        prediction = np.dot(user_features[user_idx], movie_features[:, movie_idx])
        
        # Scale prediction back to original range [0.5, 5]
        prediction = (prediction * (5 - 0.5)) + 0.5
        prediction = max(0.5, min(5, prediction))  # Clip between 0.5 and 5
        
        return f"🎬 Predicted Rating: {prediction:.2f} / 5.0"
    except Exception as e:
        return f"Error during prediction: {str(e)}"

def weighted_rating(x, m, C):
    v = x['vote_count']
    R = x['vote_average']
    return (v / (v + m) * R) + (m / (m + v) * C)

def get_top_rated_movies():
    movies = load_data()
    if movies is None:
        return "Error: Could not load movies data"
    
    C = movies['vote_average'].mean()
    m = movies['vote_count'].quantile(0.9)
    q_movies = movies.copy().loc[movies['vote_count'] >= m]
    q_movies['score'] = q_movies.apply(weighted_rating, axis=1, args=(m, C))
    q_movies = q_movies.sort_values('score', ascending=False)
    result = q_movies[['title', 'vote_count', 'vote_average', 'score']].head(10)
    formatted_result = ""
    for idx, row in result.iterrows():
        formatted_result += f"Title: {row['title']}\n"
        formatted_result += f"Vote Count: {row['vote_count']}\n"
        formatted_result += f"Vote Average: {row['vote_average']:.1f}\n"
        formatted_result += f"Score: {row['score']:.2f}\n"
        formatted_result += "-" * 50 + "\n"
    return formatted_result

def get_recommendations(title):
    movies = load_data()
    if movies is None:
        return "Error: Could not load movies data"
    
    if title not in movies['title'].values:
        return "Error: Movie not found in database"
    
    tfidf = TfidfVectorizer(stop_words='english')
    movies['overview'] = movies['overview'].fillna('')
    tfidf_matrix = tfidf.fit_transform(movies['overview'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    recommendations = movies['title'].iloc[movie_indices]
    return "\n".join([f"{i+1}. {title}" for i, title in enumerate(recommendations)])

custom_css = """
#component-0 {
    max-width: 800px !important;
    margin: auto !important;
    padding: 20px !important;
    background-color: #141414 !important;
}
.logo-container {
    text-align: center;
    margin: 0 !important;
}
.logo-image {
    max-width: 200px;
    margin: 0 !important;
}
.gradio-container {
    background-color: #141414 !important;
}
.tabs.svelte-710i53 {
    background-color: #141414 !important;
    border-bottom: 2px solid #DC1A22 !important;
    margin-bottom: 25px !important;
}
.tab-nav {
    background-color: #141414 !important;
    border: none !important;
    margin-bottom: 20px !important;
}
button.selected {
    background-color: #DC1A22 !important;
    color: white !important;
}
button {
    background-color: #DC1A22 !important;
    color: white !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 8px 16px !important;
    font-size: 14px !important;
    cursor: pointer !important;
    transition: background-color 0.3s !important;
    margin: 10px 5px !important;
}
button:hover {
    background-color: #B2070E !important;
}
.input-box, .output-box, select, textarea {
    background-color: #242424 !important;
    border: 1px solid #DC1A22 !important;
    color: white !important;
    border-radius: 4px !important;
    margin-top: 10px !important;
    margin-bottom: 15px !important;
}
label {
    color: #FFFFFF !important;
    margin-bottom: 8px !important;
}
.markdown {
    color: #FFFFFF !important;
    margin-bottom: 20px !important;
}
.tabs > div:first-child {
    border-bottom: 2px solid #DC1A22 !important;
    margin-bottom: 20px !important;
}
.tab-selected {
    color: #DC1A22 !important;
    border-bottom: 2px solid #DC1A22 !important;
}
.block {
    margin-bottom: 20px !important;
}
.row {
    margin-bottom: 15px !important;
}
"""

def create_interface():
    movies = load_data()
    if movies is None:
        return gr.Blocks().queue()
    
    # Validate logo existence
    logo_path = "nflogo.jpeg"
    if not os.path.exists(logo_path):
        print("Warning: Logo file not found, skipping logo display.")
        logo = None
    else:
        logo = logo_path
    
    with gr.Blocks(css=custom_css, theme=gr.themes.Base()) as demo:
        if logo:
            gr.Image(logo, show_label=False, container=False, scale=1, min_width=100, 
                     show_download_button=False, interactive=False, show_fullscreen_button=False)
        
        # Explicitly set text color to white using inline style
        gr.Markdown(
            "<h1 style='color: #FFFFFF;'>Movie Recommendation System</h1>",
            elem_classes="markdown"
        )
        gr.Markdown(
            "<h3 style='color: #FFFFFF;'>- By Prince Sompura</h3>",
            elem_classes="markdown"
        )
        
        with gr.Tabs() as tabs:
            with gr.Tab("Top Rated"):
                with gr.Column(scale=1):
                    demo_button = gr.Button("Show Top Rated Movies", scale=0.4)
                    gr.Markdown("     ")
                    demo_output = gr.Textbox(label="Results", lines=15)
                demo_button.click(get_top_rated_movies, inputs=[], outputs=demo_output)
            
            with gr.Tab("Find Similar"):
                with gr.Column(scale=1):
                    with gr.Row():
                        movie_dropdown = gr.Dropdown(
                            choices=movies['title'].tolist(),
                            label="Select a movie",
                            container=False,
                            scale=0.7
                        )
                        content_button = gr.Button("Get Recommendations", scale=0.3)
                    gr.Markdown("     ")
                    content_output = gr.Textbox(label="Recommended Movies", lines=10)
                content_button.click(get_recommendations, inputs=[movie_dropdown], outputs=content_output)
            
            with gr.Tab("Predict Rating"):
                with gr.Column(scale=1):
                    with gr.Row():
                        user_id = gr.Number(
                            minimum=1,
                            maximum=671,
                            value=1,
                            label="User ID",
                            scale=0.3
                        )
                        movie_id = gr.Number(
                            minimum=1,
                            maximum=100000,
                            value=302,
                            label="Movie ID",
                            scale=0.3
                        )
                        predict_button = gr.Button("Predict", scale=0.2)
                    gr.Markdown("     ")
                    predict_output = gr.Textbox(label="Predicted Rating")
                predict_button.click(predict_rating, inputs=[user_id, movie_id], outputs=predict_output)
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()