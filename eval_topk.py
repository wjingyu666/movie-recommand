# eval_topk.py
# Top-K ranking evaluation for NeuMF (Leave-One-Out + Negative Sampling)
# Just run: python eval_topk.py

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf


# =========================
# 1. 手动配置评估参数（只改这里）
# =========================
MODEL_PATH = "neumf_model.h5"
TRAIN_CSV = "train_data.csv"
TEST_CSV  = "test_data.csv"

K_LIST = [5, 10, 20]     # Top-K values
NEG_SIZE = 99            # #negative samples per user
MAX_USERS = 0            # 0 = evaluate all users
RANDOM_SEED = 42
BATCH_SIZE = 2048

# choose one positive per user from test set
# "highest_rating" or "random"
CHOOSE_POSITIVE = "highest_rating"


# =========================
# 2. Metric definitions
# =========================
def hit_rate_at_k(ranked_items: List[int], true_item: int, k: int) -> float:
    return 1.0 if true_item in ranked_items[:k] else 0.0


def ndcg_at_k(ranked_items: List[int], true_item: int, k: int) -> float:
    topk = ranked_items[:k]
    if true_item not in topk:
        return 0.0
    rank = topk.index(true_item) + 1
    return 1.0 / math.log2(rank + 1)


# =========================
# 3. Data helpers
# =========================
def build_item_feature_map(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[Dict[int, np.ndarray], List[str]]:
    df_all = pd.concat([df_train, df_test], ignore_index=True)
    feat_cols = [c for c in df_all.columns if c.startswith("feat_")]
    if not feat_cols:
        raise ValueError("No feature columns starting with 'feat_' found.")

    df_item = (
        df_all.sort_values("item_idx")
        .drop_duplicates("item_idx")[["item_idx"] + feat_cols]
    )

    item_feat_map = {
        int(row["item_idx"]): row[feat_cols].to_numpy(dtype=np.float32)
        for _, row in df_item.iterrows()
    }
    return item_feat_map, feat_cols


def build_user_histories(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Dict[int, set]:
    df_all = pd.concat(
        [df_train[["user_idx", "item_idx"]],
         df_test[["user_idx", "item_idx"]]],
        ignore_index=True,
    )
    histories: Dict[int, set] = {}
    for u, it in zip(df_all["user_idx"], df_all["item_idx"]):
        histories.setdefault(int(u), set()).add(int(it))
    return histories


def choose_one_positive_per_user(df_test: pd.DataFrame, mode: str, rng: np.random.Generator) -> Dict[int, int]:
    pos = {}
    for u, grp in df_test.groupby("user_idx"):
        if mode == "highest_rating":
            max_r = grp["rating"].max()
            row = grp[grp["rating"] == max_r].sample(
                n=1, random_state=int(rng.integers(0, 2**31 - 1))
            ).iloc[0]
        else:  # random
            row = grp.sample(
                n=1, random_state=int(rng.integers(0, 2**31 - 1))
            ).iloc[0]
        pos[int(u)] = int(row["item_idx"])
    return pos


# =========================
# 4. Model inference helper
# =========================
def predict_scores(
    model: tf.keras.Model,
    user_idx: int,
    items: List[int],
    item_feat_map: Dict[int, np.ndarray],
    feat_dim: int,
    batch_size: int
) -> np.ndarray:
    n = len(items)
    user_arr = np.full(n, user_idx, dtype=np.int32)
    item_arr = np.array(items, dtype=np.int32)

    feats = np.zeros((n, feat_dim), dtype=np.float32)
    for i, it in enumerate(items):
        if it in item_feat_map:
            feats[i] = item_feat_map[it]

    scores = model.predict(
        [user_arr, item_arr, feats],
        batch_size=batch_size,
        verbose=0
    )
    return scores.reshape(-1)


# =========================
# 5. Main evaluation
# =========================
def evaluate_topk():
    rng = np.random.default_rng(RANDOM_SEED)

    df_train = pd.read_csv(TRAIN_CSV)
    df_test  = pd.read_csv(TEST_CSV)

    item_feat_map, feat_cols = build_item_feature_map(df_train, df_test)
    feat_dim = len(feat_cols)

    all_items = sorted(
        set(df_train["item_idx"]).union(set(df_test["item_idx"]))
    )
    user_hist = build_user_histories(df_train, df_test)

    user_pos = choose_one_positive_per_user(df_test, CHOOSE_POSITIVE, rng)
    users = list(user_pos.keys())

    if MAX_USERS > 0 and len(users) > MAX_USERS:
        users = rng.choice(users, size=MAX_USERS, replace=False).tolist()

    model = tf.keras.models.load_model(MODEL_PATH)

    hr_sum   = {k: 0.0 for k in K_LIST}
    ndcg_sum = {k: 0.0 for k in K_LIST}

    for idx, u in enumerate(users, start=1):
        pos_item = user_pos[u]
        interacted = user_hist.get(u, set())

        neg_pool = [it for it in all_items if it not in interacted and it != pos_item]
        if len(neg_pool) < NEG_SIZE:
            neg_pool = [it for it in all_items if it != pos_item]

        neg_items = rng.choice(neg_pool, size=NEG_SIZE, replace=False).tolist()
        candidates = [pos_item] + neg_items

        scores = predict_scores(
            model, u, candidates, item_feat_map, feat_dim, BATCH_SIZE
        )

        ranked_items = [
            it for it, _ in sorted(
                zip(candidates, scores),
                key=lambda x: x[1],
                reverse=True
            )
        ]

        for k in K_LIST:
            hr_sum[k]   += hit_rate_at_k(ranked_items, pos_item, k)
            ndcg_sum[k] += ndcg_at_k(ranked_items, pos_item, k)

        if idx % 200 == 0:
            print(f"Evaluated {idx}/{len(users)} users")

    n_users = len(users)
    print("\n=== Top-K Recommendation Evaluation ===")
    print(f"Users evaluated: {n_users}")
    print(f"Negatives per user: {NEG_SIZE}")
    print(f"Positive selection: {CHOOSE_POSITIVE}")
    print("--------------------------------------")
    for k in K_LIST:
        print(
            f"K={k:>2d} | HR@{k} = {hr_sum[k]/n_users:.4f} "
            f"| NDCG@{k} = {ndcg_sum[k]/n_users:.4f}"
        )


# =========================
# 6. Run
# =========================
if __name__ == "__main__":
    evaluate_topk()
