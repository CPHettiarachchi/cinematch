"""
pipeline.py — CineMatch CLI Pipeline
======================================
Tests data fetch + model training from the command line.

Usage:
    python pipeline.py                    # prompts for TMDB key
    TMDB_API_KEY=xxx python pipeline.py   # uses env variable
"""

import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def banner(text):
    print(f"\n{'─' * 56}\n  {text}\n{'─' * 56}")


def main():
    api_key = os.environ.get("TMDB_API_KEY", "")
    if not api_key:
        api_key = input("Enter TMDB API key: ").strip()
    if not api_key:
        print("No key provided — exiting.")
        return

    t0 = time.time()

    # ── Step 1: Fetch TMDB data ───────────────────────────────────────────────
    banner("Step 1 · Fetch TMDB catalogue")
    from tmdb_data import _fetch_catalogue, _build_ratings
    raw = _fetch_catalogue(api_key, max_pages=3)
    print(f"  Fetched {len(raw):,} raw movies in {time.time()-t0:.1f}s")

    movies_df  = pd.DataFrame(raw).drop_duplicates("item_id").reset_index(drop=True)
    ratings_df = _build_ratings(movies_df.to_dict("records"), n_users=300)

    print(f"  Movies  : {len(movies_df):,}")
    print(f"  Users   : {ratings_df['user_id'].nunique():,}")
    print(f"  Ratings : {len(ratings_df):,}")
    sparsity = 1 - len(ratings_df) / (
        ratings_df["user_id"].nunique() * len(movies_df))
    print(f"  Sparsity: {sparsity:.1%}")

    # ── Step 2: Train models ──────────────────────────────────────────────────
    banner("Step 2 · Train models")
    from recommender import build_all_models
    m = build_all_models(movies_df, ratings_df)
    print(f"  All models ready in {time.time()-t0:.1f}s")

    # ── Step 3: Evaluate ──────────────────────────────────────────────────────
    banner("Step 3 · Evaluate (Coverage · Diversity · Novelty)")

    sample_users = list(m["matrix"].index)[:30]
    all_recs = []
    for uid in sample_users:
        ur   = m["matrix"].loc[uid]
        recs = m["hybrid"].recommend(uid, ur, movies_df, n=10)
        if recs is not None and not recs.empty:
            all_recs.append(set(recs["item_id"].tolist()))

    # Coverage: fraction of catalogue that gets recommended
    all_items = set(movies_df["item_id"])
    rec_items  = set().union(*all_recs) if all_recs else set()
    coverage   = len(rec_items) / len(all_items)

    # Diversity: avg fraction of unique genres per rec list
    def genre_diversity(item_ids):
        genres = []
        for iid in item_ids:
            row = movies_df[movies_df["item_id"] == iid]
            if not row.empty:
                genres.extend(row.iloc[0]["genres_str"].split())
        return len(set(genres)) / max(len(genres), 1)

    diversity = np.mean([genre_diversity(r) for r in all_recs]) if all_recs else 0

    # Novelty: fraction of items not in top-20% popularity
    popular_threshold = movies_df["popularity"].quantile(0.8)
    popular_ids = set(movies_df[movies_df["popularity"] >= popular_threshold]["item_id"])
    novelty = np.mean([
        len(r - popular_ids) / max(len(r), 1) for r in all_recs
    ]) if all_recs else 0

    print(f"  Coverage  : {coverage:.1%}")
    print(f"  Diversity : {diversity:.1%}")
    print(f"  Novelty   : {novelty:.1%}")

    # ── Step 4: Sample recommendations ───────────────────────────────────────
    banner("Step 4 · Sample recommendations")
    uid = m["matrix"].index[0]
    ur  = m["matrix"].loc[uid]
    print(f"  User #{uid}  |  rated {int((ur > 0).sum())} movies  "
          f"|  avg {ur[ur>0].mean():.2f}")

    recs = m["hybrid"].recommend(uid, ur, movies_df, n=8)
    if recs is not None and not recs.empty:
        print(f"\n  Hybrid top-8:")
        for _, row in recs.iterrows():
            title = row["title"][:44]
            score = row.get("hybrid_score", 0)
            year  = row.get("year", "")
            print(f"    {title:<44}  {year}  {score:.3f}")

    # ── Step 5: Cold start ────────────────────────────────────────────────────
    banner("Step 5 · Cold-start (genre: Action + Thriller)")
    cold = m["pop"].onboard(["Action", "Thriller"], n=5)
    for _, row in cold.iterrows():
        print(f"  {row['title'][:44]:<44}  ★{row.get('vote_avg',0):.1f}")

    banner(f"Pipeline complete in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
