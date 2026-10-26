"""
Unstructured Text Analysis & Automated Triage with TF-IDF
Course: Data Science Tools & Techniques

Complete Steps 1, 2, and 3.
Run from the repository root:
    python student_starter_script.py
"""

# %% Setup & Dataset Loading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Download required NLTK resources quietly
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# Load 4 real-world unstructured user discussion categories
CATEGORIES = [
    "sci.electronics",
    "rec.autos",
    "comp.sys.mac.hardware",
    "sci.med",
]

print("Loading dataset...")
raw_data = fetch_20newsgroups(
    subset="all",
    categories=CATEGORIES,
    remove=("headers", "footers", "quotes"),
)

df = pd.DataFrame(
    {
        "text": raw_data.data,
        "true_label": [raw_data.target_names[i] for i in raw_data.target],
    }
)

# Keep posts with meaningful length (> 50 characters)
df = df[df["text"].str.strip().str.len() > 50].reset_index(drop=True)
print(f"Loaded {len(df)} unstructured user posts.\n")


# %% Step 1: Text Cleaning, Tokenization & Normalization
sample_post = df["text"].iloc[0][:400]
print("--- Raw Sample Post ---")
print(sample_post)
print("-" * 40)

# TODO 1.1: Tokenize, lowercase, and keep alphabetic tokens only.
tokens = [
    w.lower()
    for w in word_tokenize(sample_post)
    if w.isalpha()
]

# TODO 1.2: Remove standard English stopwords.
stop_words = set(stopwords.words("english"))
filtered_tokens = [w for w in tokens if w not in stop_words]

# TODO 1.3: Apply stemming and verb lemmatization.
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

stemmed_tokens = [stemmer.stem(w) for w in filtered_tokens]
lemmatized_tokens = [
    lemmatizer.lemmatize(w, pos="v") for w in filtered_tokens
]

print("Filtered Tokens:  ", filtered_tokens[:12])
print("Stemmed Roots:    ", stemmed_tokens[:12])
print("Lemmatized Words: ", lemmatized_tokens[:12])


# %% Step 2: Bag of Words vs. TF-IDF Feature Extraction
# TODO 2.1: Initialize vectorizers with the requested settings.
bow_vec = CountVectorizer(
    stop_words="english",
    max_features=4000,
    min_df=5
)
tfidf_vec = TfidfVectorizer(
    stop_words="english",
    max_features=4000,
    min_df=5
)

# TODO 2.2: Fit and transform the text data.
X_bow = bow_vec.fit_transform(df["text"])
X_tfidf = tfidf_vec.fit_transform(df["text"])

# TODO 2.3: Calculate TF-IDF matrix sparsity.
total_elements = X_tfidf.shape[0] * X_tfidf.shape[1]
sparsity = 100.0 * (1.0 - (X_tfidf.nnz / total_elements))
print(f"\nTF-IDF Matrix Shape: {X_tfidf.shape} | Sparsity: {sparsity:.2f}%")

# TODO 2.4: Plot top 10 words by mean TF-IDF score.
tfidf_means = np.asarray(X_tfidf.mean(axis=0)).ravel()
top_idx = tfidf_means.argsort()[-10:]
top_words = tfidf_vec.get_feature_names_out()[top_idx]
top_scores = tfidf_means[top_idx]

plt.figure(figsize=(8, 4.5))
plt.barh(top_words, top_scores, color="teal")
plt.xlabel("Average TF-IDF Weight")
plt.title("Top 10 Most Influential Words (Baseline TF-IDF)")
plt.tight_layout()
plt.savefig("top_features.png", dpi=150)
plt.show()
print("Saved plot: top_features.png")


# %% Step 3: Automated Topic Discovery & Spatial Mapping
# TODO 3.1: Fit KMeans with the required parameters.
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(X_tfidf)

feature_names = tfidf_vec.get_feature_names_out()

# TODO 3.2: Print top 8 words per centroid and plot a heatmap of top 5 words per cluster.
print("\nTop 8 defining words per cluster:")
for cluster_id, centroid in enumerate(kmeans.cluster_centers_):
    top_indices = centroid.argsort()[-8:][::-1]
    top_terms = feature_names[top_indices]
    print(f"Cluster {cluster_id}: {', '.join(top_terms)}")

# Select the union of each cluster's top five terms for the heatmap.
top_terms_per_cluster = []
for centroid in kmeans.cluster_centers_:
    top_indices = centroid.argsort()[-5:][::-1]
    top_terms_per_cluster.extend(feature_names[top_indices])

unique_terms = list(dict.fromkeys(top_terms_per_cluster))
term_indices = [np.where(feature_names == term)[0][0] for term in unique_terms]
heatmap_data = kmeans.cluster_centers_[:, term_indices]

plt.figure(figsize=(max(10, len(unique_terms) * 0.45), 5))
sns.heatmap(
    heatmap_data,
    xticklabels=unique_terms,
    yticklabels=[f"Cluster {i}" for i in range(kmeans.n_clusters)],
    cmap="YlGnBu"
)
plt.title("Top TF-IDF Terms by Cluster")
plt.xlabel("Terms")
plt.ylabel("Cluster")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("cluster_heatmap.png", dpi=150)
plt.show()
print("Saved plot: cluster_heatmap.png")

# TODO 3.3: Project TF-IDF vectors to two dimensions and plot clusters.
svd = TruncatedSVD(n_components=2, random_state=42)
reduced_data = svd.fit_transform(X_tfidf)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    reduced_data[:, 0],
    reduced_data[:, 1],
    c=df["cluster"],
    cmap="viridis",
    alpha=0.6,
    s=10
)
plt.xlabel("SVD Component 1")
plt.ylabel("SVD Component 2")
plt.title("2D Spatial Mapping of Text Clusters")
plt.colorbar(scatter, label="Cluster")
plt.tight_layout()
plt.savefig("spatial_mapping.png", dpi=150)
plt.show()
print("Saved plot: spatial_mapping.png")

# Save a copy of the assigned cluster for convenient inspection.
df[["text", "true_label", "cluster"]].to_csv(
    "cluster_assignments.csv", index=False
)
print("Saved cluster assignments: cluster_assignments.csv")
