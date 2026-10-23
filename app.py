
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Netflix Analytics Dashboard",
    page_icon="🎬",
    layout="wide"
)

# ---------------- LIGHT THEME ----------------

st.markdown("""
<style>
/* Main dashboard */
.stApp {
    background-color: #f6f8fc !important;
    color: #202938 !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
}

[data-testid="stSidebar"] * {
    color: #202938 !important;
}

/* Headings and regular text */
.stApp h1,
.stApp h2,
.stApp h3,
.stApp p,
.stApp label,
.stApp .stMarkdown,
.stApp li {
    color: #202938 !important;
}

/* KPI cards */
[data-testid="stMetric"] {
    background-color: #ffffff !important;
    padding: 20px !important;
    border-radius: 12px !important;
    border: 1px solid #dce2ec !important;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: #475569 !important;
    opacity: 1 !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div {
    color: #111827 !important;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #334155 !important;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #e50914 !important;
    font-weight: 700 !important;
}

/* Input fields */
.stTextInput input,
.stSelectbox div,
.stMultiSelect div {
    color: #202938 !important;
}

/* Buttons */
.stButton button {
    color: #ffffff !important;
    background-color: #e50914 !important;
    border: none !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        possible_files = [
            "NETFLIX MOVIES AND TV SHOWS CLUSTERING (1).csv",
            "NETFLIX MOVIES AND TV SHOWS CLUSTERING.csv",
            "netflix_titles.csv",
        ]

        csv_path = next(
            (Path(name) for name in possible_files if Path(name).exists()),
            None
        )

        if csv_path is None:
            return None

        df = pd.read_csv(csv_path)

    df.columns = df.columns.str.strip()
    return df

with st.sidebar:
    st.header("⚙️ Dashboard Controls")
    uploaded_file = st.file_uploader(
        "Upload Netflix CSV Dataset",
        type=["csv"]
    )

    st.info("Upload your Netflix CSV if the dataset is not found.")

try:
    df = load_data(uploaded_file)
except Exception as e:
    st.error(f"Could not read the CSV file: {e}")
    st.stop()

if df is None:
    st.warning(
        "Netflix CSV Dataset not found. Upload your CSV using the sidebar."
    )
    st.stop()

# ---------------- DATA CLEANING ----------------
for col in ["type", "title", "country", "rating",
            "listed_in", "release_year", "description"]:
    if col not in df.columns:
        df[col] = None

df["type"] = df["type"].fillna("Unknown")
df["title"] = df["title"].fillna("Untitled").astype(str)
df["release_year"] = pd.to_numeric(
    df["release_year"], errors="coerce"
)

# ---------------- SIDEBAR FILTERS ----------------
with st.sidebar:
    st.subheader("🔎 Filters")

    content_types = sorted(df["type"].dropna().unique().tolist())
    selected_types = st.multiselect(
        "Content Type",
        content_types,
        default=content_types
    )

    years = df["release_year"].dropna()

    if not years.empty:
        min_year = int(years.min())
        max_year = int(years.max())

        year_range = st.slider(
            "Release Year",
            min_value=min_year,
            max_value=max_year,
            value=(min_year, max_year)
        )
    else:
        year_range = None

filtered = df[df["type"].isin(selected_types)].copy()

if year_range is not None:
    filtered = filtered[
        filtered["release_year"].between(
            year_range[0], year_range[1]
        )
    ]

# ---------------- KPI CARDS ----------------
total_titles = len(filtered)
movies = int((filtered["type"] == "Movie").sum())
tv_shows = int((filtered["type"] == "TV Show").sum())
countries = filtered["country"].dropna().astype(str).nunique()

c1, c2, c3, c4 = st.columns(4)
c1.metric("🎞️ Total Titles", f"{total_titles:,}")
c2.metric("🎬 Movies", f"{movies:,}")
c3.metric("📺 TV Shows", f"{tv_shows:,}")
c4.metric("🌍 Countries", f"{countries:,}")

st.divider()

# ---------------- CHARTS ----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🔍 Explore Titles",
    "🧠 Clustering",
    "📁 Dataset"
])

with tab1:
    left, right = st.columns(2)

    with left:
        type_counts = (
            filtered["type"]
            .value_counts()
            .rename_axis("Content Type")
            .reset_index(name="Count")
        )

        fig = px.pie(
            type_counts,
            names="Content Type",
            values="Count",
            hole=0.48,
            title="Movies vs TV Shows"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        genre_data = (
            filtered["listed_in"]
            .dropna()
            .astype(str)
            .str.split(", ")
            .explode()
            .value_counts()
            .head(12)
            .rename_axis("Genre")
            .reset_index(name="Titles")
        )

        fig = px.bar(
            genre_data,
            x="Titles",
            y="Genre",
            orientation="h",
            title="Top Genres"
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

    yearly = (
        filtered.dropna(subset=["release_year"])
        .groupby("release_year")
        .size()
        .reset_index(name="Titles")
    )

    fig = px.line(
        yearly,
        x="release_year",
        y="Titles",
        markers=True,
        title="Titles by Release Year"
    )
    st.plotly_chart(fig, use_container_width=True)

    rating_data = (
        filtered["rating"]
        .fillna("Not Rated")
        .value_counts()
        .head(12)
        .rename_axis("Rating")
        .reset_index(name="Titles")
    )

    fig = px.bar(
        rating_data,
        x="Rating",
        y="Titles",
        title="Content Ratings"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Search Netflix Titles")

    search = st.text_input(
        "Search by title",
        placeholder="Enter a movie or TV show name..."
    )

    result = filtered.copy()

    if search:
        result = result[
            result["title"].str.contains(
                search, case=False, na=False
            )
        ]

    available_cols = [
        col for col in [
            "title", "type", "director", "cast", "country",
            "release_year", "rating", "duration",
            "listed_in", "description"
        ]
        if col in result.columns
    ]

    st.write(f"Found **{len(result):,}** titles")
    st.dataframe(
        result[available_cols],
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download Filtered Titles",
        data=result.to_csv(index=False).encode("utf-8-sig"),
        file_name="netflix_filtered_titles.csv",
        mime="text/csv"
    )

with tab3:
    st.subheader("🧠 K-Means Content Clustering")

    st.write(
        "Group similar Netflix titles using their title, genre "
        "and description text."
    )

    max_clusters = min(10, len(filtered))

    if len(filtered) < 3:
        st.info("At least 3 titles are needed for clustering.")
    else:
        k = st.slider(
            "Number of clusters (K)",
            min_value=2,
            max_value=max_clusters - 1,
            value=min(5, max_clusters - 1)
        )

        if st.button("🚀 Run Clustering"):
            with st.spinner("Clustering titles..."):
                work = filtered.copy()

                text_cols = ["title", "listed_in", "description"]
                work["cluster_text"] = (
                    work[text_cols]
                    .fillna("")
                    .astype(str)
                    .agg(" ".join, axis=1)
                )

                # Limit processing time for large datasets
                if len(work) > 4000:
                    work = work.sample(
                        4000, random_state=42
                    )

                vectorizer = TfidfVectorizer(
                    stop_words="english",
                    max_features=3000,
                    min_df=1
                )

                matrix = vectorizer.fit_transform(
                    work["cluster_text"]
                )

                if matrix.shape[1] < 2 or matrix.shape[0] <= k:
                    st.error("Not enough text data for clustering.")
                else:
                    model = KMeans(
                        n_clusters=k,
                        random_state=42,
                        n_init=10
                    )

                    work["Cluster"] = model.fit_predict(matrix)

                    score = silhouette_score(
                        matrix,
                        work["Cluster"],
                        sample_size=min(1000, len(work)),
                        random_state=42
                    )

                    st.session_state["cluster_results"] = work
                    st.session_state["silhouette"] = score

    if "cluster_results" in st.session_state:
        clustered = st.session_state["cluster_results"]

        st.metric(
            "Silhouette Score",
            f'{st.session_state["silhouette"]:.4f}'
        )

        cluster_counts = (
            clustered["Cluster"]
            .value_counts()
            .sort_index()
            .rename_axis("Cluster")
            .reset_index(name="Titles")
        )

        fig = px.bar(
            cluster_counts,
            x="Cluster",
            y="Titles",
            title="Titles in Each Cluster"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            clustered[
                ["title", "type", "listed_in", "Cluster"]
            ],
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download Cluster Results",
            data=clustered.to_csv(index=False).encode("utf-8-sig"),
            file_name="Netflix_Clustering_Results.csv",
            mime="text/csv"
        )

with tab4:
    st.subheader("📁 Dataset Preview")

    st.write(f"Rows: {len(df):,} | Columns: {len(df.columns)}")

    st.dataframe(
        df.head(100),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download Full Dataset",
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name="netflix_dataset.csv",
        mime="text/csv"
    )

st.divider()
st.caption("Netflix Analytics Dashboard | Built with Python and Streamlit")
