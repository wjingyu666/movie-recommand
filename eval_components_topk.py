# eval_components_topk.py
# Component-level Top-K evaluation:
#   A) Recall-only ranking (douban_rating + log(votes))
#   B) Final ranking (same candidates, re-ranked by NeuMF)
#
# Protocol: Leave-One-Out + Negative Sampling (default 99)
# Metrics: HR@K, NDCG@K
#
# Run: python eval_components_topk.py

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf


# =========================
# 1) 配置区（只改这里）
# =========================
MODEL_PATH = "neumf_model.h5"
TRAIN_CSV = "train_data.csv"
TEST_CSV = "test_data.csv"
DOUBAN_INFO_CSV = "movie_douban_info.csv"  # needs columns: movieId, douban_rating, douban_votes

K_LIST = [5, 10, 20]
NEG_SIZE = 99
MAX_USERS = 0           # 0 = all users
RANDOM_SEED = 42
BATCH_SIZE = 2048
CHOOSE_POSITIVE = "highest_rating"  # "highest_rating" or "random"

# recall score weights (same idea as your MCP code)
W_RATING = 0.7
W_VOTES = 0.3


# =========================
# 2) Metrics
# =========================
def hit_rate_at_k(ranked_items: List[int], true_item: int, k: int) -> float:
    return 1.0 if true_item in ranked_items[:k] else 0.0


def ndcg_at_k(ranked_items: List[int], true_item: int, k: int) -> float:
    topk = ranked_items[:k]
    if true_item not in topk:
        return 0.0
    rank = topk.index(true_item) + 1  # 1-based
    return 1.0 / math.log2(rank + 1)


# =========================
# 3) Helpers: build maps
# =========================
def build_item_feature_map(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame
) -> Tuple[Dict[int, np.ndarray], List[str]]:
    """item_idx -> feature_vector, features are columns starting with feat_"""
    df_all = pd.concat([df_train, df_test], ignore_index=True)
    feat_cols = [c for c in df_all.columns if c.startswith("feat_")]
    if not feat_cols:
        raise ValueError("No feature columns starting with 'feat_' found in train/test.")

    df_item = (
        df_all.sort_values("item_idx")
        .drop_duplicates("item_idx")[["item_idx"] + feat_cols]
    )
    item_feat_map = {
        int(row["item_idx"]): row[feat_cols].to_numpy(dtype=np.float32)
        for _, row in df_item.iterrows()
    }
    return item_feat_map, feat_cols


def build_itemidx_to_movieid(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Dict[int, int]:
    """item_idx -> movieId (MovieLens id)"""
    df_all = pd.concat([df_train[["item_idx", "movieId"]], df_test[["item_idx", "movieId"]]], ignore_index=True)
    df_map = df_all.drop_duplicates("item_idx")
    return {int(r["item_idx"]): int(r["movieId"]) for _, r in df_map.iterrows()}


def build_douban_meta_map(douban_csv: str) -> Dict[int, Tuple[float, float]]:
    """
    movieId -> (douban_rating, douban_votes)
    douban_votes should be numeric
    """
    df = pd.read_csv(douban_csv)
    needed = {"movieId", "douban_rating", "douban_votes"}
    if not needed.issubset(df.columns):
        raise ValueError(f"{douban_csv} must contain columns {needed}, but got {set(df.columns)}")

    meta = {}
    for _, row in df.iterrows():
        mid = int(row["movieId"])
        try:
            rating = float(row["douban_rating"])
        except Exception:
            rating = float("nan")
        try:
            votes = float(row["douban_votes"])
        except Exception:
            votes = 0.0
        meta[mid] = (rating, votes)
    return meta


def build_user_histories(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Dict[int, set]:
    """user_idx -> set(all interacted item_idx), used to avoid sampling positives as negatives"""
    df_all = pd.concat(
        [df_train[["user_idx", "item_idx"]], df_test[["user_idx", "item_idx"]]],
        ignore_index=True,
    )
    histories: Dict[int, set] = {}
    for u, it in zip(df_all["user_idx"], df_all["item_idx"]):
        histories.setdefault(int(u), set()).add(int(it))
    return histories


def choose_one_positive_per_user(df_test: pd.DataFrame, mode: str, rng: np.random.Generator) -> Dict[int, int]:
    """Leave-One-Out: choose 1 positive item from test per user"""
    pos: Dict[int, int] = {}
    for u, grp in df_test.groupby("user_idx"):
        if mode == "highest_rating":
            max_r = grp["rating"].max()
            cand = grp[grp["rating"] == max_r]
            row = cand.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1))).iloc[0]
        elif mode == "random":
            row = grp.sample(n=1, random_state=int(rng.integers(0, 2**31 - 1))).iloc[0]
        else:
            raise ValueError(f"Unknown CHOOSE_POSITIVE: {mode}")
        pos[int(u)] = int(row["item_idx"])
    return pos


# =========================
# 4) Scoring functions
# =========================
def recall_score_for_item(
    item_idx: int,
    itemidx_to_movieid: Dict[int, int],
    douban_meta: Dict[int, Tuple[float, float]],
) -> float:
    """
    Recall-only score: W_RATING * douban_rating + W_VOTES * log(votes+1)
    If missing metadata, return very small score to push it down.
    """
    movie_id = itemidx_to_movieid.get(item_idx)
    if movie_id is None:
        return -1e9

    meta = douban_meta.get(movie_id)
    if meta is None:
        return -1e9

    rating, votes = meta
    if not np.isfinite(rating):
        return -1e9

    votes = max(float(votes), 0.0)
    return W_RATING * float(rating) + W_VOTES * math.log(votes + 1.0)


def neumf_scores(
    model: tf.keras.Model,
    user_idx: int,
    items: List[int],
    item_feat_map: Dict[int, np.ndarray],
    feat_dim: int,
    batch_size: int
) -> np.ndarray:
    """Batch NeuMF inference: scores for (user, many items)"""
    n = len(items)
    user_arr = np.full(n, user_idx, dtype=np.int32)
    item_arr = np.array(items, dtype=np.int32)

    feats = np.zeros((n, feat_dim), dtype=np.float32)
    for i, it in enumerate(items):
        f = item_feat_map.get(int(it))
        if f is not None:
            feats[i] = f

    preds = model.predict([user_arr, item_arr, feats], batch_size=batch_size, verbose=0)
    return preds.reshape(-1)


# =========================
# 5) Main evaluation
# =========================
def evaluate_components_topk():
    rng = np.random.default_rng(RANDOM_SEED)

    df_train = pd.read_csv(TRAIN_CSV)
    df_test = pd.read_csv(TEST_CSV)

    # basic checks
    for col in ["user_idx", "item_idx", "movieId"]:
        if col not in df_train.columns or col not in df_test.columns:
            raise ValueError(f"train/test must contain column '{col}'")

    if "rating" not in df_test.columns:
        raise ValueError("test must contain column 'rating'")

    # maps
    item_feat_map, feat_cols = build_item_feature_map(df_train, df_test)
    feat_dim = len(feat_cols)

    itemidx_to_movieid = build_itemidx_to_movieid(df_train, df_test)
    douban_meta = build_douban_meta_map(DOUBAN_INFO_CSV)

    user_hist = build_user_histories(df_train, df_test)
    user_pos = choose_one_positive_per_user(df_test, CHOOSE_POSITIVE, rng)

    users = list(user_pos.keys())
    if MAX_USERS > 0 and len(users) > MAX_USERS:
        users = rng.choice(users, size=MAX_USERS, replace=False).tolist()

    all_items = sorted(set(df_train["item_idx"]).union(set(df_test["item_idx"])))
    model = tf.keras.models.load_model(MODEL_PATH)

    ks = sorted(K_LIST)

    # accumulators for two components
    hr_recall = {k: 0.0 for k in ks}
    ndcg_recall = {k: 0.0 for k in ks}
    hr_final = {k: 0.0 for k in ks}
    ndcg_final = {k: 0.0 for k in ks}

    for idx, u in enumerate(users, start=1):
        pos_item = user_pos[u]
        interacted = user_hist.get(u, set())

        # negative sampling: items user never interacted with
        neg_pool = [it for it in all_items if it not in interacted and it != pos_item]
        if len(neg_pool) < NEG_SIZE:
            neg_pool = [it for it in all_items if it != pos_item]

        negs = rng.choice(neg_pool, size=NEG_SIZE, replace=False).tolist()
        candidates = [pos_item] + negs

        # A) recall-only ranking
        recall_scores = np.array([
            recall_score_for_item(it, itemidx_to_movieid, douban_meta)
            for it in candidates
        ], dtype=np.float32)
        ranked_recall = [it for it, _ in sorted(zip(candidates, recall_scores), key=lambda x: x[1], reverse=True)]

        # B) final ranking (NeuMF reranking on same candidates)
        final_scores = neumf_scores(model, u, candidates, item_feat_map, feat_dim, BATCH_SIZE)
        ranked_final = [it for it, _ in sorted(zip(candidates, final_scores), key=lambda x: x[1], reverse=True)]

        for k in ks:
            hr_recall[k] += hit_rate_at_k(ranked_recall, pos_item, k)
            ndcg_recall[k] += ndcg_at_k(ranked_recall, pos_item, k)

            hr_final[k] += hit_rate_at_k(ranked_final, pos_item, k)
            ndcg_final[k] += ndcg_at_k(ranked_final, pos_item, k)

        if idx % 200 == 0:
            print(f"Evaluated {idx}/{len(users)} users...")

    n = len(users)
    print("\n=== Component-level Top-K Evaluation (same candidates) ===")
    print(f"Users evaluated: {n}")
    print(f"Negatives per user: {NEG_SIZE}")
    print(f"Positive selection: {CHOOSE_POSITIVE}")
    print("---------------------------------------------------------")
    for k in ks:
        a_hr = hr_recall[k] / n
        a_nd = ndcg_recall[k] / n
        b_hr = hr_final[k] / n
        b_nd = ndcg_final[k] / n
        print(
            f"K={k:>2d} | "
            f"RecallOnly HR@{k}={a_hr:.4f}, NDCG@{k}={a_nd:.4f}  ||  "
            f"Final HR@{k}={b_hr:.4f}, NDCG@{k}={b_nd:.4f}  ||  "
            f"ΔHR={b_hr-a_hr:+.4f}, ΔNDCG={b_nd-a_nd:+.4f}"
        )


if __name__ == "__main__":
    evaluate_components_topk()