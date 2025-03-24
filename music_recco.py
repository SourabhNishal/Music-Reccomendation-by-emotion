import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load and preprocess the data
songs = pd.read_csv('updated_songdata.csv')
songs = songs.sample(n=5000).reset_index(drop=True)
songs['text'] = songs['text'].str.replace(r'\n', '')

# Create the TF-IDF matrix
tfidf = TfidfVectorizer(analyzer='word', stop_words='english')
lyrics_matrix = tfidf.fit_transform(songs['text'])

# Compute cosine similarities
cosine_similarities = cosine_similarity(lyrics_matrix)

# Create a dictionary to hold the similarities
similarities = {}
for i in range(len(cosine_similarities)):
    similar_indices = cosine_similarities[i].argsort()[:-50:-1]
    similarities[songs['song'].iloc[i]] = [(cosine_similarities[i][x], songs['song'][x], songs['artist'][x]) for x in similar_indices][1:]

class ContentBasedRecommender:
    def __init__(self, matrix):
        self.matrix_similar = matrix

    def _print_message(self, song, recom_song):
        rec_items = len(recom_song)
        print(f'The {rec_items} recommended songs for "{song}" are:')
        for i in range(rec_items):
            print(f"Number {i+1}:")
            print(f"{recom_song[i][1]} by {recom_song[i][2]}")
            print("--------------------")

    def recommend(self, song):
        number_songs = 5
        if song in self.matrix_similar:
            recom_song = self.matrix_similar[song][:number_songs]
            self._print_message(song=song, recom_song=recom_song)
        else:
            print(f'Song "{song}" not found in the dataset.')

# Instantiate the recommender
recommender = ContentBasedRecommender(similarities)

# Function to get recommendations based on user input
def get_recommendations(song_name: str):
    recommender.recommend(song_name)

# Example usage
song_name = input("Enter the song name: ")
get_recommendations(song_name)
