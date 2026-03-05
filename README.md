# 🎬 Content-Based Movie Recommendation Engine using NLP

An end-to-end **Movie Recommendation Engine** that suggests movies based on their content such as **genre, storyline, director, and cast**.

This project uses Natural Language Processing techniques to convert movie descriptions into numerical vectors using **TF-IDF** and finds similar movies using **Cosine Similarity**.

The system is deployed with an interactive web interface using **Streamlit**, allowing users to search movies and receive recommendations instantly.

---

# 📌 Problem Statement

With thousands of movies released every year across multiple streaming platforms, users often struggle to find movies that match their preferences.

Traditional search systems depend on exact keywords or genres, which fail to capture deeper relationships between movies such as similar themes, storytelling styles, or cast.

The goal of this project is to build a **content-based recommendation system** that analyzes movie metadata and recommends movies that are most similar based on their descriptive features.

---

# 🎯 Project Objectives

The objectives of this project are:

* Clean and preprocess movie metadata
* Analyze movie trends using exploratory data analysis
* Extract meaningful features from movie descriptions
* Convert textual information into numerical vectors
* Build a **content-based recommendation engine**
* Develop an interactive web interface for users

---

# 📂 Dataset

Dataset: **IMDB Movie Dataset (1951–2023)**

The dataset contains metadata of thousands of movies.

### Key Features

| Feature    | Description                |
| ---------- | -------------------------- |
| movie_name | Title of the movie         |
| year       | Release year               |
| genre      | Movie genre                |
| overview   | Movie storyline or summary |
| director   | Director of the movie      |
| cast       | Main actors                |

---

# 🔄 Data Science Pipeline

The project follows a structured data science workflow:

```
Problem Definition
        ↓
Data Collection
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Text Vectorization (TF-IDF)
        ↓
Similarity Calculation (Cosine Similarity)
        ↓
Recommendation Engine
        ↓
Streamlit Web Application
```

---

# 🧹 Data Cleaning

Several preprocessing steps were applied to prepare the dataset:

* Removed unnecessary columns
* Cleaned the **year column** using regular expressions
* Converted year values to numeric format
* Removed missing values
* Filled missing text fields such as genre, overview, director, and cast

Example:

```python
df["year"] = pd.to_numeric(df["year"], errors="coerce")
df = df.dropna(subset=["year"])
```

---

# 📊 Exploratory Data Analysis (EDA)

EDA was performed to understand patterns in the dataset.

Key analyses include:

* Movies released per year
* Top movie genres
* Most frequent directors
* Most frequent actors

Visualization tools used:

* Pandas
* Matplotlib
* Seaborn

---

# ⚙ Feature Engineering

Important textual attributes were combined to create a rich feature representation.

```
combined_features = genre + overview + director + cast
```

This combined text helps capture the **complete context of each movie**.

---

# 🤖 Model / Algorithm

The project uses a **Content-Based Filtering Approach**.

Instead of learning user preferences, the system recommends movies based on their similarity to other movies.

## 1. TF-IDF Vectorization

TF-IDF converts movie text descriptions into numerical vectors.

It assigns higher weight to important words and lower weight to common words.

Example text:

```
Action movies contain fights, explosions and heroic characters
```

Each word is converted into a **numerical vector representation**.

---

## 2. Cosine Similarity

Cosine similarity measures the similarity between two movies.

Similarity values range between:

```
0 → Completely different
1 → Identical
```

Movies with higher similarity scores are recommended.

---

# 🧠 Recommendation Engine

The system supports two recommendation methods.

### 1️⃣ Recommendation by Movie Title

The system finds movies that are most similar to a given movie.

Example:

```python
recommend_movies("Avatar")
```

---

### 2️⃣ Recommendation by Movie Description

Users can input a genre and storyline to discover similar movies.

Example:

```python
recommend_from_description(
    "Action Thriller",
    "A brave police officer fights a gang to save the city",
    5
)
```

---

# 🌐 Streamlit Web Application

An interactive web interface was built using **Streamlit**.

Users can:

* Enter a movie title to find similar movies
* Enter a movie description to get recommendations
* Explore movie suggestions instantly

Run the application locally:

```
streamlit run app.py
```

The application will start at:

```
http://localhost:8501
```

---

# 📁 Project Structure

```
Content-Based-Movie-Recommendation-Engine-using-NLP
│
├── IMDB-Movie-Dataset(2023-1951).csv 
│
├── movie_recommendation_pipeline.py
│
├── Movie_Recomendation_System.ipynb
│
├── app.py
│
└── README.md
```

---

# 🛠 Technologies Used

| Technology   | Purpose              |
| ------------ | -------------------- |
| Python       | Programming language |
| Pandas       | Data analysis        |
| NumPy        | Numerical computing  |
| Matplotlib   | Visualization        |
| Seaborn      | Visualization        |
| Scikit-learn | Machine learning     |
| Streamlit    | Web application      |

---

# 📈 Results

The system successfully recommends movies with similar:

* genre
* story themes
* narrative style
* directors
* actors

Example output:

| Recommended Movie | Year | Director   |
| ----------------- | ---- | ---------- |
| Movie A           | 2016 | Director X |
| Movie B           | 2018 | Director Y |
| Movie C           | 2020 | Director Z |

---

# 🚀 Future Improvements

Potential enhancements include:

* Add movie posters using external APIs
* Deploy the app to cloud platforms
* Implement collaborative filtering
* Use deep learning-based recommendation models
* Add user ratings and personalized recommendations

---

# 📌 Conclusion

This project demonstrates how **Natural Language Processing techniques can be used to build a practical recommendation system**.

By combining **TF-IDF vectorization and cosine similarity**, the system effectively identifies relationships between movies and recommends relevant content.

The addition of a **Streamlit web application** enables users to interact with the recommendation engine through a simple and intuitive interface.

---

# 👨‍💻 Author

**Vivek Pradip Sonawane**

Computer Engineering Student
Aspiring Data Analyst / Data Scientist
