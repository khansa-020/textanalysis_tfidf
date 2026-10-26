# Part 2: Problem-Solving Experience — Answers

These answers explain the expected behavior of the diagnostics. Exact cluster IDs, keyword rankings, document weights, and category overlaps depend on the dataset version and the results produced when `student_starter_script.py` is run.

## Challenge 1: Conversational Filler & Numeric Noise

TF-IDF is commonly defined as

\[
\operatorname{TFIDF}(t,d)=\operatorname{TF}(t,d)\times\log\left(\frac{N}{\operatorname{DF}(t)}\right).
\]

A word such as `know` can still receive a substantial score because it is not automatically removed by the standard English stopword list unless it is explicitly included. If it appears in 25% of posts, its inverse-document-frequency factor is approximately \(\log(1/0.25)=\log(4)\) (about 1.386 with the natural logarithm), before accounting for the term-frequency definition and any vector normalization. Thus, appearing in many posts does not by itself make its weight zero; it is down-weighted relative to rarer terms, while a custom stopword list can remove it entirely.

For the cleaner model, use the requested alphabetic token pattern, a maximum document frequency, and custom conversational stopwords:

```python
custom_stopwords = sorted(
    set(stopwords.words("english"))
    | {"know", "like", "just", "think", "does", "thanks", "use"}
)

clean_tfidf_vec = TfidfVectorizer(
    token_pattern=r"\b[a-zA-Z]{3,}\b",
    stop_words=custom_stopwords,
    max_df=0.5,
    min_df=5,
    max_features=4000
)
X_clean = clean_tfidf_vec.fit_transform(df["text"])

clean_kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clean_clusters = clean_kmeans.fit_predict(X_clean)
```

The token pattern excludes tokens shorter than three alphabetic characters, which removes standalone numeric tokens and many short fragments. `max_df=0.5` also removes terms appearing in more than half of the posts. After refitting, cluster keywords should generally be less dominated by conversational filler and more focused on subject-specific terms. The exact top words must be taken from the printed output of the run rather than guessed in advance.

## Challenge 2: Hardware Vocabulary Overlap

Use the cross-tabulation to compare the original categories with the unsupervised cluster assignments:

```python
print(pd.crosstab(df["true_label"], df["cluster"]))
```

The two hardware-related categories to inspect are `sci.electronics` and `comp.sys.mac.hardware`. They may overlap because both discuss computer/electronic equipment and technical troubleshooting. Candidate shared terms to check in the misclustered posts include **`computer`**, **`drive`**, and **`card`**; the actual words present in the most informative misclustered examples should be verified from the data before reporting them as findings.

Bigrams can capture short phrases whose meanings are more specific than individual words. For example, `hard drive`, `video card`, or `Mac OS` may help distinguish computer hardware discussions from general electronics, provided those phrases occur often enough in the corpus. To test this, fit a second vectorizer with `ngram_range=(1, 2)`, refit KMeans, and compare its cross-tabulation and top cluster terms with the unigram baseline. Bigrams can improve separation, but they do not guarantee it; the result should be judged from the observed clustering output.

Example experiment:

```python
bigram_vec = TfidfVectorizer(
    stop_words="english",
    max_features=4000,
    min_df=5,
    ngram_range=(1, 2)
)
X_bigram = bigram_vec.fit_transform(df["text"])

bigram_kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
bigram_clusters = bigram_kmeans.fit_predict(X_bigram)

print(pd.crosstab(df["true_label"], bigram_clusters))
```

## Challenge 3: Rant Length Bias (BoW vs. TF-IDF)

The Euclidean norm is

\[
\|x\|_2=\sqrt{\sum_i x_i^2}.
\]

Calculate the norms for the longest and shortest posts using the matrices created in Step 2:

```python
lengths = df["text"].str.len()
longest_id = lengths.idxmax()
shortest_id = lengths.idxmin()

for name, doc_id in [("Longest", longest_id), ("Shortest", shortest_id)]:
    bow_norm = np.linalg.norm(X_bow[doc_id].toarray())
    tfidf_norm = np.linalg.norm(X_tfidf[doc_id].toarray())
    print(
        f"{name} post (row {doc_id}; {lengths.iloc[doc_id]} characters): "
        f"BoW L2 norm={bow_norm:.4f}, TF-IDF L2 norm={tfidf_norm:.4f}"
    )
```

Raw Bag of Words vectors contain term counts, so a long post often has a much larger vector norm than a short post. KMeans uses Euclidean distance, so differences in document length can dominate distances and pull long posts together even when their topics differ. Scikit-Learn's `TfidfVectorizer` uses L2 normalization by default (`norm="l2"`), making each non-empty document vector have a norm of approximately 1; this reduces length-driven scale differences and gives terms more influence according to their TF-IDF weights. It does not eliminate every clustering problem, but it is a more suitable baseline for comparing documents of different lengths.

## Challenge 4: Regulated Compliance Explainability

Use the baseline TF-IDF matrix and its matching vectorizer to print the five highest-weighted non-zero features for document 412:

```python
def explain_document(doc_id, matrix, vectorizer, top_n=5):
    if not 0 <= doc_id < matrix.shape[0]:
        raise IndexError(
            f"doc_id must be between 0 and {matrix.shape[0] - 1}"
        )

    row = matrix[doc_id].toarray().ravel()
    feature_names = vectorizer.get_feature_names_out()
    nonzero_indices = np.flatnonzero(row)

    top_indices = nonzero_indices[
        np.argsort(row[nonzero_indices])[-top_n:][::-1]
    ]
    results = [(feature_names[i], float(row[i])) for i in top_indices]

    print(f"Top {len(results)} non-zero TF-IDF terms for document {doc_id}:")
    for term, weight in results:
        print(f"{term}: {weight:.6f}")

    return results


# Run after X_tfidf and tfidf_vec have been created in Step 2:
explain_document(412, X_tfidf, tfidf_vec, top_n=5)
```

**Three-sentence compliance justification:** TF-IDF provides a reproducible, inspectable baseline because each document's explanation can list the exact terms and numerical weights that influenced its representation. Unlike a black-box LLM response, these feature weights can be recalculated from the same corpus and vectorizer settings and independently checked by an auditor. TF-IDF alone does not establish clinical meaning, causality, or regulatory compliance, so it should support—not replace—clinical review, documented validation, and appropriate governance.
