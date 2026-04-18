"""
src/recommender.py
ML recommendation engine — works purely on TMDB data.
No MovieLens files required.

Models:
  ContentBasedRecommender   — TF-IDF on genres + cosine similarity
  CollaborativeFilter       — mean-centered user-user cosine similarity
  SVDRecommender            — TruncatedSVD matrix factorisation
  HybridRecommender         — weighted ensemble (CB + CF + SVD)
  PopularityRecommender     — Bayesian-average popularity (cold start)
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# CONTENT-BASED
# ─────────────────────────────────────────────────────────────────────────────
class ContentBasedRecommender:
    """
    TF-IDF on genre strings + cosine similarity matrix.
    Also uses year and language as weak signals.
    """
    def __init__(self):
        self.tfidf    = TfidfVectorizer(token_pattern=r"\b\w+\b", min_df=1)
        self.sim      = None
        self.movies   = None
        self.id_index = {}   # item_id → row index

    def fit(self, movies_df: pd.DataFrame):
        self.movies = movies_df.reset_index(drop=True)
        self.id_index = {row["item_id"]: i
                         for i, row in self.movies.iterrows()}
        # Enrich text: genres + language + decade
        def enrich(row):
            parts = [str(row.get("genres_str",""))]
            lang  = str(row.get("language",""))
            if lang: parts.append(lang)
            year  = str(row.get("year",""))[:3]   # decade prefix
            if year: parts.append(f"decade_{year}0s")
            return " ".join(parts)

        corpus = self.movies.apply(enrich, axis=1).fillna("unknown")
        mat    = self.tfidf.fit_transform(corpus)
        self.sim = cosine_similarity(mat)
        return self

    def recommend_by_id(self, item_id, n: int = 10) -> pd.DataFrame:
        if item_id not in self.id_index:
            return pd.DataFrame()
        idx    = self.id_index[item_id]
        scores = list(enumerate(self.sim[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
        out = []
        for i, sc in scores:
            row = self.movies.iloc[i]
            out.append({
                "item_id":          row["item_id"],
                "tmdb_id":          row.get("tmdb_id", row["item_id"]),
                "title":            row["title"],
                "year":             row.get("year",""),
                "genres":           row.get("genres_str",""),
                "poster_url":       row.get("poster_url",""),
                "vote_avg":         row.get("vote_avg",0),
                "similarity_score": round(sc, 4),
            })
        return pd.DataFrame(out)

    def recommend_for_profile(self, liked_ids: list, disliked_ids: list = None,
                               n: int = 10) -> pd.DataFrame:
        """
        Aggregate similarity from a set of liked movies.
        Penalise disliked movies (optional).
        """
        acc = np.zeros(len(self.movies))
        for iid in liked_ids:
            if iid in self.id_index:
                acc += self.sim[self.id_index[iid]]
        if disliked_ids:
            for iid in disliked_ids:
                if iid in self.id_index:
                    acc -= self.sim[self.id_index[iid]] * 0.5
        # Zero out source movies
        for iid in liked_ids + (disliked_ids or []):
            if iid in self.id_index:
                acc[self.id_index[iid]] = 0

        top = np.argsort(acc)[::-1][:n]
        out = []
        for i in top:
            row = self.movies.iloc[i]
            out.append({
                "item_id":    row["item_id"],
                "tmdb_id":    row.get("tmdb_id", row["item_id"]),
                "title":      row["title"],
                "year":       row.get("year",""),
                "genres":     row.get("genres_str",""),
                "poster_url": row.get("poster_url",""),
                "vote_avg":   row.get("vote_avg",0),
                "score":      round(float(acc[i]), 4),
            })
        return pd.DataFrame(out)

    def recommend_for_user(self, user_ratings: pd.Series, n: int = 10):
        liked = user_ratings[user_ratings >= 3.5].index.tolist()
        if not liked:
            liked = user_ratings.nlargest(3).index.tolist()
        return self.recommend_for_profile(liked, n=n)


# ─────────────────────────────────────────────────────────────────────────────
# COLLABORATIVE FILTERING
# ─────────────────────────────────────────────────────────────────────────────
class CollaborativeFilter:
    """User-User CF with mean-centered cosine similarity."""

    def __init__(self, k: int = 25):
        self.k      = k
        self.sim    = None
        self.matrix = None
        self.means  = None

    def fit(self, matrix: pd.DataFrame):
        self.matrix = matrix
        M    = matrix.values.astype(float)
        cnt  = (M != 0).sum(axis=1)
        self.means = np.where(cnt > 0, M.sum(axis=1) / np.maximum(cnt, 1), 0)
        C    = M.copy()
        for i in range(M.shape[0]):
            mask = M[i] != 0
            if mask.any():
                C[i, mask] -= self.means[i]
        self.sim = cosine_similarity(C)
        return self

    def predict(self, user_id, item_id) -> float:
        users = list(self.matrix.index)
        items = list(self.matrix.columns)
        if user_id not in users or item_id not in items:
            return float(self.means.mean()) if self.means is not None else 3.0
        u = users.index(user_id)
        i = items.index(item_id)
        ir   = self.matrix.iloc[:, i].values
        mask = (ir != 0) & (np.arange(len(users)) != u)
        sims = self.sim[u] * mask
        top  = [j for j in np.argsort(sims)[::-1][:self.k] if ir[j] != 0]
        if not top:
            return float(self.means[u])
        num = sum(self.sim[u][j] * (ir[j] - self.means[j]) for j in top)
        den = sum(abs(self.sim[u][j]) for j in top) + 1e-8
        return float(np.clip(self.means[u] + num / den, 1, 5))

    def recommend(self, user_id, movies_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        if user_id not in self.matrix.index:
            return pd.DataFrame()
        ur    = self.matrix.loc[user_id]
        preds = [(iid, self.predict(user_id, iid))
                 for iid in ur[ur == 0].index]
        preds.sort(key=lambda x: x[1], reverse=True)
        out = []
        for iid, pred in preds[:n]:
            row = movies_df[movies_df["item_id"] == iid]
            if not row.empty:
                r = row.iloc[0]
                out.append({"item_id": iid,
                            "tmdb_id":     r.get("tmdb_id", iid),
                            "title":       r["title"],
                            "year":        r.get("year",""),
                            "genres":      r.get("genres_str",""),
                            "poster_url":  r.get("poster_url",""),
                            "vote_avg":    r.get("vote_avg",0),
                            "predicted_rating": round(pred, 2)})
        return pd.DataFrame(out)


# ─────────────────────────────────────────────────────────────────────────────
# SVD
# ─────────────────────────────────────────────────────────────────────────────
class SVDRecommender:
    """Truncated SVD matrix factorisation — R ≈ U Σ Vᵀ"""

    def __init__(self, n_factors: int = 50):
        self.n_factors = n_factors
        self.svd       = TruncatedSVD(n_components=n_factors, random_state=42)
        self.scaler    = MinMaxScaler(feature_range=(1, 5))
        self.matrix    = None
        self._pred     = None

    def fit(self, matrix: pd.DataFrame):
        self.matrix = matrix
        M   = matrix.values.astype(float)
        UF  = self.svd.fit_transform(M)
        raw = UF @ self.svd.components_
        self._pred = self.scaler.fit_transform(
            raw.flatten().reshape(-1, 1)
        ).reshape(raw.shape)
        ev = self.svd.explained_variance_ratio_.sum()
        print(f"✓ SVD {self.n_factors} factors — {ev:.1%} variance explained")
        return self

    def predict(self, user_id, item_id) -> float:
        users = list(self.matrix.index)
        items = list(self.matrix.columns)
        if user_id not in users or item_id not in items:
            return 3.0
        return float(np.clip(
            self._pred[users.index(user_id), items.index(item_id)], 1, 5))

    def recommend(self, user_id, movies_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        users = list(self.matrix.index)
        items = list(self.matrix.columns)
        if user_id not in users:
            return pd.DataFrame()
        u  = users.index(user_id)
        ur = self.matrix.loc[user_id]
        unrated = [items.index(iid) for iid in ur[ur == 0].index if iid in items]
        scored  = sorted([(items[i], float(self._pred[u, i])) for i in unrated],
                         key=lambda x: x[1], reverse=True)[:n]
        out = []
        for iid, sc in scored:
            row = movies_df[movies_df["item_id"] == iid]
            if not row.empty:
                r = row.iloc[0]
                out.append({"item_id": iid,
                            "tmdb_id":    r.get("tmdb_id", iid),
                            "title":      r["title"],
                            "year":       r.get("year",""),
                            "genres":     r.get("genres_str",""),
                            "poster_url": r.get("poster_url",""),
                            "vote_avg":   r.get("vote_avg",0),
                            "predicted_rating": round(sc, 2)})
        return pd.DataFrame(out)


# ─────────────────────────────────────────────────────────────────────────────
# HYBRID
# ─────────────────────────────────────────────────────────────────────────────
class HybridRecommender:
    """Weighted score fusion: CB + CF + SVD"""

    def __init__(self, cb, cf, svd, weights=(0.25, 0.35, 0.40)):
        self.cb, self.cf, self.svd = cb, cf, svd
        self.w = weights

    def recommend(self, user_id, user_ratings: pd.Series,
                  movies_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        def norm(df, col):
            if df is None or df.empty or col not in df.columns:
                return {}
            v  = df[col].values.reshape(-1, 1)
            nv = MinMaxScaler().fit_transform(v).flatten()
            return dict(zip(df["item_id"], nv))

        cb_r  = self.cb.recommend_for_user(user_ratings, n=n*3)
        cf_r  = self.cf.recommend(user_id, movies_df, n=n*3)
        svd_r = self.svd.recommend(user_id, movies_df, n=n*3)

        cs = norm(cb_r,  "score")
        fs = norm(cf_r,  "predicted_rating")
        ss = norm(svd_r, "predicted_rating")

        all_ids = set(cs) | set(fs) | set(ss)
        fused   = {iid: self.w[0]*cs.get(iid,0) +
                        self.w[1]*fs.get(iid,0) +
                        self.w[2]*ss.get(iid,0)
                   for iid in all_ids}

        for iid in user_ratings[user_ratings > 0].index:
            fused.pop(iid, None)

        top = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:n]
        out = []
        for iid, sc in top:
            row = movies_df[movies_df["item_id"] == iid]
            if not row.empty:
                r = row.iloc[0]
                out.append({
                    "item_id":      iid,
                    "tmdb_id":      r.get("tmdb_id", iid),
                    "title":        r["title"],
                    "year":         r.get("year",""),
                    "genres":       r.get("genres_str",""),
                    "poster_url":   r.get("poster_url",""),
                    "vote_avg":     r.get("vote_avg",0),
                    "hybrid_score": round(sc, 4),
                })
        return pd.DataFrame(out)


# ─────────────────────────────────────────────────────────────────────────────
# POPULARITY (cold-start)
# ─────────────────────────────────────────────────────────────────────────────
class PopularityRecommender:
    """
    Bayesian-average ranking — same formula as IMDb's weighted rating.
    Works for brand-new users with zero history.
    """
    def __init__(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame):
        self.movies = movies_df
        stats = ratings_df.groupby("item_id")["rating"].agg(["mean","count"])
        C = stats["count"].quantile(0.5)
        v = stats["mean"].mean()
        stats["score"] = (stats["count"]/(stats["count"]+C)*stats["mean"] +
                          C/(stats["count"]+C)*v)
        self.pop = stats.reset_index()

    def recommend(self, n: int = 12, genre: str = None,
                  min_year: int = None, max_year: int = None,
                  language: str = None) -> pd.DataFrame:
        df = self.movies.merge(self.pop, on="item_id", how="left")
        df["score"] = df["score"].fillna(df.get("vote_avg", pd.Series(dtype=float)))
        df["score"] = df["score"].fillna(0)

        if genre:
            df = df[df["genres_str"].str.contains(genre, case=False, na=False)]
        if min_year:
            df = df[df["year"] >= str(min_year)]
        if max_year:
            df = df[df["year"] <= str(max_year)]
        if language:
            df = df[df["language"] == language]

        return df.nlargest(n, "score")[
            ["item_id","tmdb_id","title","year","genres_str",
             "poster_url","vote_avg","score"]
        ].rename(columns={"genres_str":"genres"})

    def onboard(self, genres: list, n: int = 10) -> pd.DataFrame:
        frames = [self.recommend(n=n*2, genre=g) for g in genres]
        if not frames:
            return self.recommend(n=n)
        return (pd.concat(frames)
                .drop_duplicates("item_id")
                .nlargest(n, "score"))


# ─────────────────────────────────────────────────────────────────────────────
# MODEL BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_all_models(movies_df: pd.DataFrame,
                     ratings_df: pd.DataFrame) -> dict:
    """
    Fit all models and return them in a dict.
    Called once after TMDB dataset is ready.
    """
    matrix = ratings_df.pivot_table(
        index="user_id", columns="item_id", values="rating"
    ).fillna(0)

    cb  = ContentBasedRecommender().fit(movies_df)
    cf  = CollaborativeFilter(k=25).fit(matrix)
    svd = SVDRecommender(n_factors=50).fit(matrix)
    hyb = HybridRecommender(cb, cf, svd)
    pop = PopularityRecommender(movies_df, ratings_df)

    return dict(
        cb=cb, cf=cf, svd=svd, hybrid=hyb, pop=pop,
        matrix=matrix,
        movies=movies_df, ratings=ratings_df,
        n_movies=len(movies_df),
        n_users=ratings_df["user_id"].nunique(),
        n_ratings=len(ratings_df),
    )
