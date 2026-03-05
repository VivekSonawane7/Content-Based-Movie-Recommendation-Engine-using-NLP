"""
Movie Recommendation System Pipeline
Author: Vivek Pradip Sonawane

Pipeline:
Problem Statement
↓
Data Cleaning
↓
Exploratory Data Analysis
↓
Feature Engineering
↓
Vectorization (TF-IDF)
↓
Similarity Calculation (Cosine Similarity)
↓
Recommendation Engine
"""

# ============================================
# 1 Import Libraries
# ============================================

import pandas as pd
import numpy as np
import re
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================
# 2 Load Dataset
# ============================================

def load_data(file_path):
    df = pd.read_csv(file_path)
    print("Dataset Loaded Successfully")
    print("Shape:", df.shape)
    return df


# ============================================
# 3 Data Cleaning
# ============================================

def clean_data(df):

    # Clean year column
    df["year"] = df["year"].astype(str).str.strip()

    df["year"] = df["year"].apply(
        lambda x: re.search(r"\b(19\d{2}|20\d{2})\b", x).group(0)
        if re.search(r"\b(19\d{2}|20\d{2})\b", x)
        else np.nan
    )

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"]).reset_index(drop=True)
    df["year"] = df["year"].astype(int)

    # Remove unnecessary column
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    # Fill missing text columns
    df["genre"] = df["genre"].fillna("")
    df["overview"] = df["overview"].fillna("")
    df["director"] = df["director"].fillna("")
    df["cast"] = df["cast"].fillna("")

    print("Data Cleaning Completed")

    return df


# ============================================
# 4 Exploratory Data Analysis
# ============================================

def perform_eda(df):

    print("\nMovies per Year")

    movies_per_year = df["year"].value_counts().sort_index()

    plt.figure(figsize=(12,6))
    movies_per_year.plot(kind="bar", color="skyblue")
    plt.title("Movies Released per Year")
    plt.xlabel("Year")
    plt.ylabel("Count")
    plt.show()

    # Top Genres
    all_genres = []

    for genres in df["genre"]:
        all_genres.extend([g.strip() for g in genres.split(",")])

    genre_counts = Counter(all_genres)

    pd.Series(genre_counts).sort_values(ascending=False).head(10).plot(
        kind="bar", color="orange"
    )

    plt.title("Top 10 Genres")
    plt.ylabel("Count")
    plt.show()

    # Top Directors
    df["director"].value_counts().head(10).plot(kind="bar", color="green")
    plt.title("Top Directors")
    plt.show()

    # Top Actors
    all_actors = []

    for cast_list in df["cast"]:
        all_actors.extend([actor.strip() for actor in cast_list.split(",")])

    actor_counts = Counter(all_actors)

    pd.Series(actor_counts).sort_values(ascending=False).head(10).plot(
        kind="bar", color="purple"
    )

    plt.title("Top Actors")
    plt.show()


# ============================================
# 5 Feature Engineering
# ============================================

def create_features(df):

    df["combined_features"] = (
        df["genre"] + " " +
        df["overview"] + " " +
        df["director"] + " " +
        df["cast"]
    )

    print("Feature Engineering Completed")

    return df


# ============================================
# 6 Vectorization
# ============================================

def vectorize_text(df):

    vectorizer = TfidfVectorizer(stop_words="english")

    tfidf_matrix = vectorizer.fit_transform(df["combined_features"])

    print("TF-IDF Vectorization Completed")

    return vectorizer, tfidf_matrix


# ============================================
# 7 Similarity Matrix
# ============================================

def compute_similarity(tfidf_matrix):

    similarity_matrix = cosine_similarity(tfidf_matrix)

    print("Similarity Matrix Created")

    return similarity_matrix


# ============================================
# 8 Recommendation by Movie Title
# ============================================

def recommend_movies(movie_title, df, similarity_matrix, top_n=5):

    movie_title = movie_title.lower()

    if movie_title not in df["movie_name"].str.lower().values:
        return "Movie not found in dataset"

    movie_index = df[df["movie_name"].str.lower() == movie_title].index[0]

    similarity_scores = list(enumerate(similarity_matrix[movie_index]))

    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    top_movies = similarity_scores[1:top_n+1]

    movie_indices = [i[0] for i in top_movies]

    return df[["movie_name","year","director"]].iloc[movie_indices]


# ============================================
# 9 Recommendation by Description
# ============================================

def recommend_from_description(genre, overview, vectorizer, tfidf_matrix, df, top_n=5):

    query = genre + " " + overview

    query_vec = vectorizer.transform([query])

    similarity_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_indices = similarity_scores.argsort()[::-1][:top_n]

    return df[["movie_name","year","director"]].iloc[top_indices]


# ============================================
# 10 Main Pipeline
# ============================================

def build_model():

    df = load_data("IMDB-Movie-Dataset(2023-1951).csv")

    df = clean_data(df)

    df = create_features(df)

    vectorizer, tfidf_matrix = vectorize_text(df)

    similarity_matrix = compute_similarity(tfidf_matrix)

    return df, vectorizer, tfidf_matrix, similarity_matrix