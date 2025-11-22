from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def cluster_subqueries(subqueries: List[str], k: int = 3) -> Tuple[List[str], Dict[int, List[str]]]:
    """Cluster sub-queries and return representative queries and full clusters.

    Prefer sentence-transformers (MiniLM) embeddings when available (better semantic clustering).
    Falls back to TF-IDF + KMeans when sentence-transformers is not installed or if embedding step fails.
    """
    cleaned = [sq.strip() for sq in subqueries if sq and sq.strip()]
    if not cleaned:
        return [], {}
    if len(cleaned) <= k:
        return cleaned, {i: [sq] for i, sq in enumerate(cleaned)}

    # Try embedding-based clustering first (MiniLM). If that fails, use TF-IDF fallback.
    try:
        try:
            from sentence_transformers import SentenceTransformer

            # prefer a compact MiniLM model for speed; if unavailable, fallback will occur
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
            emb_model = SentenceTransformer(model_name)
            X_emb = emb_model.encode(cleaned, convert_to_numpy=True, show_progress_bar=False)
            X = X_emb
            use_embeddings = True
        except Exception:
            use_embeddings = False

        if not use_embeddings:
            # TF-IDF fallback
            vectorizer = TfidfVectorizer(stop_words="english")
            X = vectorizer.fit_transform(cleaned)

        k = min(k, len(cleaned))
        # KMeans works with dense arrays; if X is sparse, convert to dense for centroids
        if hasattr(X, "toarray"):
            X_dense = X.toarray()
        else:
            X_dense = X if isinstance(X, np.ndarray) else np.asarray(X)

        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X_dense)
        clusters: Dict[int, List[int]] = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(label, []).append(idx)

        representatives: List[str] = []
        cluster_texts: Dict[int, List[str]] = {}
        centroids = kmeans.cluster_centers_
        for label, indices in clusters.items():
            cluster_texts[label] = [cleaned[i] for i in indices]
            # pick the query closest to centroid by cosine similarity
            centroid = centroids[label].reshape(1, -1)
            sims = cosine_similarity(X_dense[indices], centroid).flatten()
            best_local = int(np.argmax(sims))
            best_idx = indices[best_local]
            representatives.append(cleaned[best_idx])
        return representatives, cluster_texts
    except Exception:
        # Worst-case: return top-k original cleaned queries
        return cleaned[:k], {i: [sq] for i, sq in enumerate(cleaned[:k])}
