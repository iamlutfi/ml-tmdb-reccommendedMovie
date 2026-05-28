# 🎬 Movie Recommendation System
A content-based movie recommendation system built on the TMDB 5000 dataset. Given a movie title, it finds the most similar films by analyzing genres, keywords, cast, director, and plot overview then ranks them using a weighted quality score to surface results that are both relevant and well-regarded.

## How It Works
```
Movie title input
      ↓
Fuzzy match → tolerant of typos, finds closest title in dataset
      ↓
TF-IDF vectorization → converts combined text features into numerical vectors
      ↓
Cosine similarity → finds top 20 most similar movies
      ↓
Re-rank with weighted rating (IMDB formula) → penalizes low-vote films
      ↓
Final score = (similarity × 0.7) + (weighted_rating × 0.3)
      ↓
Top 10 recommendations with poster
```

## Project Structure
```
├── data/
│   ├── raw/                            # Original TMDB 5000 dataset (not tracked)
│   └── preprocessed/                   # Cleaned & processed data (not tracked)
├── model/                              # Saved cosine similarity matrix (not tracked)
├── notebooks/
│   ├── 00_EDA.ipynb                    # Exploratory data analysis
│   ├── 01_Preprocessing.ipynb          # Cleaning, parsing, stemming, feature engineering
│   └── 02_ContentBasedModelling.ipynb  # TF-IDF, cosine similarity, recommendations
├── src/
│   └── app.py                          # Streamlit application
├── pyproject.toml                      # Project dependencies (uv)
├── uv.lock                             # Locked dependency versions
└── .gitignore
```

## Features
- **Content-based filtering** — similarity based on genres, keywords, cast, director, and overview
- **Fuzzy matching** — typo-tolerant title search via `difflib`
- **NLTK stemming** — applied selectively to overview only, preserving names and genres
- **Weighted rating** — IMDB formula to filter films with insufficient votes
- **Movie posters** — fetched live via TMDB API

## Tech Stack
- **Python** — core language
- **uv** — fast Python package manager & project environment tool
- **Pandas, NumPy** — data manipulation
- **Scikit-learn** — TF-IDF vectorization & cosine similarity
- **NLTK** — Porter Stemmer for text normalization
- **Streamlit** — web application
- **TMDB API** — movie poster retrieval

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
# Install uv (if not already installed)
pip install uv
```

```bash
# Clone the repository
git clone https://github.com/iamlutfi/ml-tmdb-recommended-movie.git
cd ml-tmdb-recommended-movie

# Create virtual environment & install all dependencies
uv sync
```

Create a `.env` file in the root directory:
```
TMDB_API_KEY=your_api_key_here
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) and place both CSV files in `data/raw/`. Then run the notebooks in order (`00` → `01` → `02`) to generate the preprocessed data and model files.

Dataset link: https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

```bash
# Run the app
uv run streamlit run src/app.py
```

## Key Concepts
| Concept | Description |
|---|---|
| TF-IDF | Weighs words by importance across all documents |
| Cosine Similarity | Measures directional similarity between movie vectors |
| Weighted Rating | Balances rating quality with vote count (IMDB formula) |
| Porter Stemmer | Reduces words to base form — applied to overview only |
| Fuzzy Matching | Finds closest title match to handle typos |
