# 🎵 Spotify Music Trends Analyser

**What Makes a Song Popular?**

An interactive data analytics web app analysing 114,000+ Spotify tracks across 114 genres to uncover what audio features drive popularity.

Built by **Vishak Edakkattuparambil Biju** 

🚀 **[Live App →](https://your-app-url.streamlit.app)** *(update this link after deployment)*

---

## Key Findings

- **Loudness** is the strongest positive predictor of popularity (r = +0.35)
- **Instrumentalness** is the strongest negative predictor (r = −0.30)
- Only ~5% of songs achieve Very High popularity (80+/100)
- Explicit songs consistently outperform clean songs in average popularity

## Features

- 📈 **Popularity Analysis** — distribution charts and category breakdowns
- 🎸 **Genre Insights** — top genres, danceability vs popularity scatter
- 🔊 **Audio Features** — interactive correlation heatmap, feature vs popularity with OLS trendline
- 🏆 **Top Songs & Artists** — leaderboards and live search

## Tech Stack

- Python · Pandas · NumPy · Matplotlib · Seaborn · Plotly · Streamlit

## Dataset

[Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset) — Kaggle (114,000 tracks, 114 genres)

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

> Place `spotify_track_dataset.csv` in the same folder as `app.py` before running.
