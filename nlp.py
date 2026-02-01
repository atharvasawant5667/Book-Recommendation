import streamlit as st
import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Audible Insights 📚",
    layout="wide"
)

st.title("📚 Audible Insights: Intelligent Book Recommendation System")
st.caption("Content-based NLP recommendations using TF-IDF, Cosine Similarity & Clustering")

# =====================================================
# HELPER FUNCTIONS (MUST BE FIRST)
# =====================================================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# =====================================================
# DATA LOADING & CLEANING
# =====================================================

@st.cache_data
def load_and_clean_data():
    df1 = pd.read_excel("Audible_Catlog.xlsx")
    df2 = pd.read_excel("Audible_Catlog_Advanced_Features.xlsx")

    df = pd.merge(
        df1,
        df2,
        on=["Book Name", "Author", "Rating", "Number of Reviews", "Price"],
        how="inner"
    )

    # Fix ratings
    df["Rating"] = df["Rating"].replace(-1, np.nan)
    df["Rating"].fillna(df["Rating"].mean(), inplace=True)

    # Descriptions
    df["Description"].fillna("No description available", inplace=True)
    df["clean_description"] = df["Description"].apply(clean_text)

    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df

# =====================================================
# TF-IDF + COSINE SIMILARITY
# =====================================================

@st.cache_data
def build_similarity(corpus):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000
    )
    matrix = vectorizer.fit_transform(corpus)
    similarity = cosine_similarity(matrix)
    return similarity, matrix, vectorizer

# =====================================================
# CLUSTERING
# =====================================================

@st.cache_data
def build_clusters(_matrix, k=6):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    return model.fit_predict(_matrix)

def generate_cluster_names(matrix, labels, vectorizer, top_n=3):
    feature_names = np.array(vectorizer.get_feature_names_out())
    cluster_names = {}

    for cluster_id in np.unique(labels):
        indices = np.where(labels == cluster_id)[0]
        mean_tfidf = matrix[indices].mean(axis=0)
        top_features = np.asarray(mean_tfidf).flatten().argsort()[-top_n:]
        keywords = feature_names[top_features]
        cluster_names[cluster_id] = " / ".join(keywords).title()

    return cluster_names

# =====================================================
# MAIN PIPELINE (ORDER IS CRITICAL)
# =====================================================

df = load_and_clean_data()

cosine_sim, tfidf_matrix, tfidf_vectorizer = build_similarity(
    df["clean_description"]
)

# 🔥 THIS LINE CREATES THE CLUSTER COLUMN
df["Cluster"] = build_clusters(tfidf_matrix)

cluster_names = generate_cluster_names(
    tfidf_matrix,
    df["Cluster"].values,
    tfidf_vectorizer
)

df["Cluster Name"] = df["Cluster"].map(cluster_names)

# =====================================================
# RECOMMENDATION FUNCTION
# =====================================================

def recommend_books(book_name, top_n=5):
    if book_name not in df["Book Name"].values:
        return pd.DataFrame()

    idx = df.index[df["Book Name"] == book_name][0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    indices = [i[0] for i in scores]

    return df.loc[indices, ["Book Name", "Author", "Rating"]]

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("🔍 Navigation")

section = st.sidebar.radio(
    "Choose Section",
    ["EDA", "Book Recommendation", "Topic-Based Recommendation", "Hidden Gems"]
)

# =====================================================
# EDA
# =====================================================

if section == "EDA":
    st.header("📊 Exploratory Data Analysis")

    st.subheader("Rating Distribution")
    st.bar_chart(df["Rating"].value_counts().sort_index())

    st.subheader("Books per Topic Cluster")
    st.bar_chart(df["Cluster Name"].value_counts())

# =====================================================
# CONTENT-BASED RECOMMENDATION
# =====================================================

elif section == "Book Recommendation":
    st.header("📖 Content-Based Recommendation")

    book = st.selectbox(
        "Select a book you like",
        sorted(df["Book Name"].unique())
    )

    if st.button("Recommend"):
        recs = recommend_books(book)
        st.dataframe(recs, use_container_width=True)

# =====================================================
# CLUSTER-BASED (NAMED TOPICS)
# =====================================================

elif section == "Topic-Based Recommendation":
    st.header("🎯 Topic-Based Recommendation (ML Generated)")

    selected_label = st.selectbox(
        "Select Topic",
        sorted(cluster_names.items()),
        format_func=lambda x: f"Topic {x[0]}: {x[1]}"
    )

    cluster_id = selected_label[0]

    top_books = (
        df[df["Cluster"] == cluster_id]
        .sort_values(by="Rating", ascending=False)
        .head(10)
    )

    st.dataframe(
        top_books[["Book Name", "Author", "Rating"]],
        use_container_width=True
    )

# =====================================================
# HIDDEN GEMS
# =====================================================

elif section == "Hidden Gems":
    st.header("💎 Hidden Gems")

    gems = df[
        (df["Rating"] >= 4.5) &
        (df["Number of Reviews"] < 100)
    ].sort_values(by="Rating", ascending=False)

    st.dataframe(
        gems[["Book Name", "Author", "Rating", "Number of Reviews"]].head(10),
        use_container_width=True
    )

# =====================================================
# FOOTER
# =====================================================

st.markdown("---")
st.caption(
    "Built using NLP (TF-IDF), Cosine Similarity & K-Means Clustering with Streamlit"
)
