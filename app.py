from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the trained k-NN model and genre columns
knn_model = joblib.load('knn_model.joblib')
genre_columns = joblib.load('genre_columns.joblib')
df = pd.read_csv('processed_netflix_titles.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    selected_genres = request.json.get('genres')
    
    # Create an input vector for the selected genres
    input_vector = np.zeros(len(genre_columns))
    for genre in selected_genres:
        if genre in genre_columns:
            input_vector[genre_columns.index(genre)] = 1
    
    # Find the nearest neighbors
    distances, indices = knn_model.kneighbors([input_vector])
    
    # Get the recommended movie titles
    recommended_movies = df.iloc[indices[0]]['title'].tolist()
    
    return jsonify(recommended_movies)

if __name__ == '__main__':
    app.run(debug=True)
