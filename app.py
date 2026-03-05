import streamlit as st
from movie_recommendation_pipeline import (
    build_model,
    recommend_movies,
    recommend_from_description
)

# Page Title
st.title("🎬 Movie Recommendation System")

st.write("Find movies similar to your favorite ones.")

# Load model
df, vectorizer, tfidf_matrix, similarity_matrix = build_model()

# ---------------------------------------
# Option 1: Recommend by Movie Title
# ---------------------------------------

st.header("Recommend by Movie Title")

movie_name = st.text_input("Enter Movie Name")

if st.button("Recommend Movies"):

    results = recommend_movies(
        movie_name,
        df,
        similarity_matrix,
        top_n=5
    )

    st.write(results)

# ---------------------------------------
# Option 2: Recommend by Description
# ---------------------------------------

st.header("Recommend by Description")

genre = st.text_input("Genre Example: Action Thriller")

overview = st.text_area("Movie Story Description")

if st.button("Find Similar Movies"):

    results = recommend_from_description(
        genre,
        overview,
        vectorizer,
        tfidf_matrix,
        df,
        top_n=5
    )

    st.write(results)