from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

try:
    import numpy as np
except ImportError as _np_err:
    raise ImportError(
        "numpy is required for semantic_analysis (pip install numpy)"
    ) from _np_err


_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\\-]+")

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "our",
    "out",
    "that",
    "the",
    "their",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "were",
    "with",
    "you",
    "your",
}


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom = np.where(denom == 0, 1.0, denom)
    return x / denom


def _pca_2d(x: np.ndarray) -> np.ndarray:
    # Center then project via SVD.
    x0 = x - np.mean(x, axis=0, keepdims=True)
    # If dim < 2, pad.
    if x0.shape[1] == 1:
        return np.hstack([x0, np.zeros((x0.shape[0], 1), dtype=x0.dtype)])
    _, _, vt = np.linalg.svd(x0, full_matrices=False)
    return x0 @ vt[:2].T


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def _keywords_fallback(texts: Sequence[str], top_n: int = 6) -> list[str]:
    counts: dict[str, int] = {}
    for t in texts:
        for w in _WORD_RE.findall(t.lower()):
            if w in _STOPWORDS:
                continue
            counts[w] = counts.get(w, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    return [w for w, _c in ranked[:top_n]]


def _keywords_tfidf(texts: Sequence[str], top_n: int = 6) -> list[str]:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception:
        return _keywords_fallback(texts, top_n=top_n)

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=2000)
    mat = vec.fit_transform(list(texts))
    if mat.shape[1] == 0:
        return []
    scores = np.asarray(mat.mean(axis=0)).ravel()
    terms = np.asarray(vec.get_feature_names_out())
    top_idx = np.argsort(scores)[::-1][:top_n]
    return [str(terms[i]) for i in top_idx if scores[i] > 0]


def label_from_keywords(keywords: Sequence[str]) -> str:
    kws = [k for k in keywords if k]
    if not kws:
        return "Misc"
    return " / ".join(kws[:3])


@dataclass(frozen=True)
class TopicPoint:
    id: str
    x: float
    y: float
    cluster_id: int
    probability: float | None = None


@dataclass(frozen=True)
class TopicCluster:
    cluster_id: int
    size: int
    label: str
    keywords: list[str]
    ids: list[str]


@dataclass(frozen=True)
class TopicDiscoveryResult:
    algorithm: str
    clusters: list[TopicCluster]
    noise_ids: list[str]
    points: list[TopicPoint]


def discover_topics(
    *,
    ids: Sequence[str],
    texts: Sequence[str],
    embeddings: Sequence[Sequence[float]],
    algorithm: str = "auto",  # auto|hdbscan|kmeans
    n_clusters: int = 12,
    min_cluster_size: int = 5,
    umap_n_components: int = 5,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.0,
    random_state: int = 42,
    top_keywords: int = 6,
) -> TopicDiscoveryResult:
    if len(ids) != len(texts) or len(ids) != len(embeddings):
        raise ValueError("ids, texts, and embeddings must have the same length")
    if not ids:
        return TopicDiscoveryResult(
            algorithm="none", clusters=[], noise_ids=[], points=[]
        )

    x = np.asarray(embeddings, dtype=np.float32)
    x = _normalize_rows(x)

    # 2D projection for visualization
    coords_2d = None
    try:
        import umap  # type: ignore

        reducer_2d = umap.UMAP(
            n_components=2,
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric="cosine",
            random_state=random_state,
        )
        coords_2d = reducer_2d.fit_transform(x)
    except Exception:
        coords_2d = _pca_2d(x)

    # Clustering space
    cluster_space = x
    used_umap = False
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(
            n_components=max(2, int(umap_n_components)),
            n_neighbors=umap_n_neighbors,
            min_dist=umap_min_dist,
            metric="cosine",
            random_state=random_state,
        )
        cluster_space = reducer.fit_transform(x)
        used_umap = True
    except Exception:
        cluster_space = x

    labels: np.ndarray
    probs: np.ndarray | None = None
    algo = algorithm.lower().strip()
    _umap_suffix = "+umap" if used_umap else ""

    if algo in {"auto", "hdbscan"}:
        try:
            import hdbscan  # type: ignore

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=max(2, int(min_cluster_size)),
                min_samples=None,
                metric="euclidean",
            )
            labels = clusterer.fit_predict(cluster_space)
            probs = getattr(clusterer, "probabilities_", None)
            algo = f"hdbscan{_umap_suffix}"
        except Exception:
            if algo == "hdbscan":
                raise
            algo = f"kmeans{_umap_suffix}"
            labels, probs = _kmeans(
                cluster_space, n_clusters=n_clusters, random_state=random_state
            )
    elif algo == "kmeans":
        algo = f"kmeans{_umap_suffix}"
        labels, probs = _kmeans(
            cluster_space, n_clusters=n_clusters, random_state=random_state
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # Build clusters + auto labels.
    clusters: list[TopicCluster] = []
    noise_ids: list[str] = []

    by_label: dict[int, list[int]] = {}
    for i, lab in enumerate(labels.tolist()):
        if lab == -1:
            noise_ids.append(ids[i])
            continue
        by_label.setdefault(int(lab), []).append(i)

    for lab, idxs in sorted(by_label.items(), key=lambda kv: len(kv[1]), reverse=True):
        cluster_texts = [texts[i] for i in idxs]
        keywords = _keywords_tfidf(cluster_texts, top_n=top_keywords)
        label = label_from_keywords(keywords)
        clusters.append(
            TopicCluster(
                cluster_id=int(lab),
                size=len(idxs),
                label=label,
                keywords=keywords,
                ids=[ids[i] for i in idxs],
            )
        )

    points: list[TopicPoint] = []
    for i, item_id in enumerate(ids):
        p = float(probs[i]) if probs is not None else None
        points.append(
            TopicPoint(
                id=str(item_id),
                x=float(coords_2d[i, 0]),
                y=float(coords_2d[i, 1]),
                cluster_id=int(labels[i]),
                probability=p,
            )
        )

    return TopicDiscoveryResult(
        algorithm=algo,
        clusters=clusters,
        noise_ids=noise_ids,
        points=points,
    )


def _kmeans(
    x: np.ndarray, *, n_clusters: int, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray | None]:
    """Numpy-only k-means (Lloyd) for environments without scikit-learn."""
    rng = np.random.default_rng(seed=random_state)
    n = x.shape[0]
    k = max(1, min(int(n_clusters), n))

    # Init centroids from random points.
    centroids = x[rng.choice(n, size=k, replace=False)].copy()

    labels = np.zeros(n, dtype=np.int32)
    for _ in range(40):
        # Assign
        dists = np.sum((x[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dists, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # Update
        for j in range(k):
            mask = labels == j
            if not np.any(mask):
                centroids[j] = x[rng.integers(0, n)]
            else:
                centroids[j] = np.mean(x[mask], axis=0)

    return labels, None


def find_semantic_duplicates(
    *,
    ids: Sequence[str],
    embeddings: Sequence[Sequence[float]],
    similarity_threshold: float = 0.92,
) -> list[list[str]]:
    if len(ids) != len(embeddings):
        raise ValueError("ids and embeddings must have the same length")
    n = len(ids)
    if n == 0:
        return []

    x = np.asarray(embeddings, dtype=np.float32)
    x = _normalize_rows(x)

    uf = _UnionFind(n)

    # Brute force but streaming, avoids building an NxN matrix.
    for i in range(n):
        sims = x[i] @ x.T
        js = np.nonzero(sims >= float(similarity_threshold))[0]
        for j in js.tolist():
            if j <= i:
                continue
            uf.union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        r = uf.find(i)
        groups.setdefault(r, []).append(i)

    out: list[list[str]] = []
    for idxs in groups.values():
        if len(idxs) <= 1:
            continue
        out.append([str(ids[i]) for i in idxs])

    # Largest groups first.
    out.sort(key=len, reverse=True)
    return out


__all__ = [
    "discover_topics",
    "find_semantic_duplicates",
    "label_from_keywords",
    "TopicDiscoveryResult",
    "TopicCluster",
    "TopicPoint",
]
