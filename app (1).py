import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="Spotify Music Trends Analyser",
    page_icon="🎵",
    layout="wide"
)

# ============================================
# HEADER
# ============================================
st.title("🎵 Spotify Music Trends Analyser")
st.markdown("### What Makes a Song Popular?")
st.markdown("""
*Analysing 114,000+ Spotify tracks across 114 genres*
**By Vishak Edakkattuparambil Biju — MSc Data Science, University of East London**
""")
st.divider()

# ============================================
# LOAD DATA
# ============================================
@st.cache_data
def load_data():
    df = pd.read_csv("spotify_track_dataset.csv")
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    df = df.dropna(subset=["artists", "album_name", "track_name"])
    df = df.sort_values("popularity", ascending=False)
    df = df.drop_duplicates(subset=["track_name", "artists"], keep="first")
    df["duration_min"] = (df["duration_ms"] / 60000).round(2)
    df["mood"] = pd.cut(
        df["valence"],
        bins=[0, 0.33, 0.66, 1.0],
        labels=["Sad/Dark", "Neutral", "Happy/Positive"]
    )
    return df

df = load_data()

# ============================================
# SIDEBAR FILTERS
# ============================================
st.sidebar.title("🎛️ Filters")
st.sidebar.markdown("Customise your analysis:")

all_genres = sorted(df["track_genre"].unique())
selected_genres = st.sidebar.multiselect(
    "Select Genres",
    options=all_genres,
    default=["pop", "rock", "hip-hop", "k-pop", "metal"]
)

min_pop, max_pop = st.sidebar.slider(
    "Popularity Range",
    min_value=0, max_value=100,
    value=(0, 100)
)

explicit_filter = st.sidebar.radio(
    "Song Type",
    options=["All Songs", "Clean Only", "Explicit Only"]
)

# Apply filters
filtered_df = df.copy()
if selected_genres:
    filtered_df = filtered_df[filtered_df["track_genre"].isin(selected_genres)]
filtered_df = filtered_df[
    (filtered_df["popularity"] >= min_pop) &
    (filtered_df["popularity"] <= max_pop)
]
if explicit_filter == "Clean Only":
    filtered_df = filtered_df[filtered_df["explicit"] == False]
elif explicit_filter == "Explicit Only":
    filtered_df = filtered_df[filtered_df["explicit"] == True]

# ============================================
# KEY METRICS ROW
# ============================================
st.subheader("📊 Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("🎵 Total Songs", f"{len(filtered_df):,}")
col2.metric("👨‍🎤 Unique Artists", f"{filtered_df['artists'].nunique():,}")
col3.metric("⭐ Avg Popularity", f"{filtered_df['popularity'].mean():.1f}/100")
col4.metric("🎼 Genres", f"{filtered_df['track_genre'].nunique()}")

st.divider()

# ============================================
# TABS
# ============================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Popularity Analysis",
    "🎸 Genre Insights",
    "🔊 Audio Features",
    "🏆 Top Songs & Artists"
])

# ============================================
# TAB 1 — POPULARITY ANALYSIS
# ============================================
with tab1:
    st.subheader("📈 How Popular Are Songs?")
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(
            filtered_df, x="popularity", nbins=50,
            title="Distribution of Song Popularity",
            color_discrete_sequence=["#1DB954"],
            labels={"popularity": "Popularity Score"}
        )
        fig.add_vline(
            x=filtered_df["popularity"].mean(),
            line_dash="dash", line_color="red",
            annotation_text=f"Mean: {filtered_df['popularity'].mean():.1f}"
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        bins = [0, 20, 40, 60, 80, 100]
        labels = ["Very Low (0-20)", "Low (21-40)", "Medium (41-60)",
                  "High (61-80)", "Very High (81-100)"]
        filtered_df["pop_category"] = pd.cut(
            filtered_df["popularity"], bins=bins, labels=labels
        )
        cat_counts = filtered_df["pop_category"].value_counts().sort_index()
        fig2 = px.bar(
            x=cat_counts.index, y=cat_counts.values,
            title="Songs by Popularity Category",
            color=cat_counts.values,
            color_continuous_scale="RdYlGn",
            labels={"x": "Category", "y": "Number of Songs"}
        )
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    very_high = len(filtered_df[filtered_df["popularity"] > 80])
    pct = very_high / len(filtered_df) * 100 if len(filtered_df) > 0 else 0
    st.info(f"💡 **Key Insight:** Only {very_high:,} songs ({pct:.1f}%) achieve Very High popularity (80+). Getting to the top is extremely rare!")

# ============================================
# TAB 2 — GENRE INSIGHTS
# ============================================
with tab2:
    st.subheader("🎸 Genre Performance")
    genre_stats = filtered_df.groupby("track_genre").agg(
        avg_popularity=("popularity", "mean"),
        song_count=("track_name", "count"),
        avg_danceability=("danceability", "mean"),
        avg_energy=("energy", "mean")
    ).round(2).reset_index()

    col1, col2 = st.columns(2)

    with col1:
        top_n = st.slider("Show Top N Genres", 5, 20, 10)
        top_genres = genre_stats.nlargest(top_n, "avg_popularity")
        fig3 = px.bar(
            top_genres.sort_values("avg_popularity"),
            x="avg_popularity", y="track_genre",
            orientation="h",
            title=f"Top {top_n} Genres by Popularity",
            color="avg_popularity",
            color_continuous_scale="Greens",
            labels={"avg_popularity": "Avg Popularity", "track_genre": "Genre"}
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        fig4 = px.scatter(
            genre_stats,
            x="avg_danceability", y="avg_popularity",
            size="song_count", color="avg_energy",
            hover_name="track_genre",
            title="Genre: Danceability vs Popularity",
            color_continuous_scale="RdYlGn",
            labels={
                "avg_danceability": "Avg Danceability",
                "avg_popularity": "Avg Popularity",
                "avg_energy": "Avg Energy"
            }
        )
        st.plotly_chart(fig4, use_container_width=True)

# ============================================
# TAB 3 — AUDIO FEATURES
# ============================================
with tab3:
    st.subheader("🔊 Audio Feature Analysis")
    col1, col2 = st.columns(2)

    with col1:
        audio_features = [
            "popularity", "danceability", "energy", "loudness",
            "speechiness", "acousticness", "instrumentalness",
            "liveness", "valence", "tempo"
        ]
        corr_matrix = filtered_df[audio_features].corr()
        fig5, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix, annot=True, fmt=".2f",
            cmap="RdYlGn", center=0, square=True,
            linewidths=0.5, ax=ax, annot_kws={"size": 9}
        )
        ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig5)

    with col2:
        feature = st.selectbox(
            "Choose a feature to compare with Popularity:",
            options=["danceability", "energy", "loudness", "valence",
                     "tempo", "acousticness", "speechiness", "instrumentalness"]
        )
        sample = filtered_df.sample(min(3000, len(filtered_df)), random_state=42)
        fig6 = px.scatter(
            sample, x=feature, y="popularity",
            color="track_genre",
            title=f"{feature.title()} vs Popularity",
            opacity=0.6,
            trendline="ols",
            labels={feature: feature.title(), "popularity": "Popularity Score"}
        )
        fig6.update_traces(showlegend=False)
        st.plotly_chart(fig6, use_container_width=True)

        corr_val = filtered_df[feature].corr(filtered_df["popularity"])
        if corr_val > 0.05:
            st.success(f"✅ {feature.title()} has a POSITIVE effect on popularity (r={corr_val:.3f})")
        elif corr_val < -0.05:
            st.error(f"❌ {feature.title()} has a NEGATIVE effect on popularity (r={corr_val:.3f})")
        else:
            st.warning(f"⚠️ {feature.title()} has MINIMAL effect on popularity (r={corr_val:.3f})")

# ============================================
# TAB 4 — TOP SONGS & ARTISTS
# ============================================
with tab4:
    st.subheader("🏆 Top Performers")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎵 Top 10 Most Popular Songs")
        top_songs = filtered_df.nlargest(10, "popularity")[
            ["track_name", "artists", "popularity", "track_genre"]
        ].reset_index(drop=True)
        top_songs.index += 1
        st.dataframe(top_songs, use_container_width=True)

    with col2:
        st.markdown("#### 👨‍🎤 Top 10 Most Popular Artists")
        artist_pop = filtered_df.groupby("artists")["popularity"].mean()
        artist_count = filtered_df.groupby("artists")["artists"].count()
        top_artists = artist_pop[artist_count >= 2].nlargest(10)
        fig7 = px.bar(
            x=top_artists.values, y=top_artists.index,
            orientation="h",
            title="Top 10 Artists by Avg Popularity",
            color=top_artists.values,
            color_continuous_scale="Greens",
            labels={"x": "Avg Popularity", "y": "Artist"}
        )
        fig7.update_layout(showlegend=False)
        st.plotly_chart(fig7, use_container_width=True)

    st.divider()
    st.markdown("#### 🔍 Search Any Song or Artist")
    search = st.text_input("Type a song name or artist...")
    if search:
        results = filtered_df[
            filtered_df["track_name"].str.contains(search, case=False, na=False) |
            filtered_df["artists"].str.contains(search, case=False, na=False)
        ][["track_name", "artists", "popularity", "track_genre", "danceability", "energy"]].head(10)

        if len(results) > 0:
            st.dataframe(results, use_container_width=True)
        else:
            st.warning("No songs found. Try a different search!")

# ============================================
# FOOTER
# ============================================
st.divider()
st.markdown("""
<div style="text-align: center; color: grey;">
    <p>🎵 Spotify Music Trends Analyser |
    Built by Vishak Edakkattuparambil Biju |
    MSc Data Science, University of East London | 2026</p>
</div>
""", unsafe_allow_html=True)
