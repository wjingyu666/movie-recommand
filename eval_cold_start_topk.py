# eval_cold_start_topk.py
# Cold-start专项评估（不使用NeuMF，只评估冷启动热门策略）
# Protocol: Leave-One-Out (1 positive from test per user) + Top-K ranking
# Metrics: HR@K, NDCG@K
#
# Run: python eval_cold_start_topk.py

import math
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# =========================
# 1) 配置区（只改这里）
# =========================
TRAIN_CSV = "train_data.csv"
TEST_CSV = "test_data.csv"
DOUBAN_INFO_CSV = "movie_douban_info.csv"  # needs columns: movieId, douban_rating, douban_votes

K_LIST = [5, 10, 20]
RANDOM_SEED = 42

# 冷启动判定：训练集中交互数 <= 阈值
COLD_THRESHOLD = 5

# 从 test 里为每个用户选 1 个正样本
# "highest_rating" or "random"
CHOOSE_POSITIVE = "highest_rating"

# 热门（冷启动）打分：同你系统的召回思想
W_RATING = 0.7
W_VOTES = 0.3

# 冷启动推荐时是否过滤掉该用户在 train 中交互过的物品
# True 更符合“不要推荐已看过”，也更利于离线评估
FILTER_TRAIN_INTERACTIONS = True


# =========================
# 2) 指标
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
# 3) 数据与映射构建
# =========================
def choose_one_positive_per_user(df_test: pd.DataFrame, mode: str, rng: np.random.Generator) -> Dict[int, int]:
    """Leave-One-Out: 每个用户从 test 里选 1 个正样本 item_idx"""
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


def build_itemidx_to_movieid(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Dict[int, int]:
    """item_idx -> movieId"""
    df_all = pd.concat(
        [df_train[["item_idx", "movieId"]], df_test[["item_idx", "movieId"]]],
        ignore_index=True
    ).drop_duplicates("item_idx")
    return {int(r["item_idx"]): int(r["movieId"]) for _, r in df_all.iterrows()}


def build_douban_meta_map(douban_csv: str) -> Dict[int, Tuple[float, float]]:
    """movieId -> (douban_rating, douban_votes)"""
    df = pd.read_csv(douban_csv)
    need = {"movieId", "douban_rating", "douban_votes"}
    if not need.issubset(df.columns):
        raise ValueError(f"{douban_csv} must contain columns {need}, but got {set(df.columns)}")

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


def build_user_train_hist(df_train: pd.DataFrame) -> Dict[int, set]:
    """user_idx -> set(item_idx) in TRAIN (用于过滤已看过)"""
    hist: Dict[int, set] = {}
    for u, it in zip(df_train["user_idx"], df_train["item_idx"]):
        hist.setdefault(int(u), set()).add(int(it))
    return hist


def build_user_train_counts(df_train: pd.DataFrame) -> Dict[int, int]:
    """user_idx -> #interactions in TRAIN"""
    vc = df_train["user_idx"].value_counts()
    return {int(u): int(c) for u, c in vc.items()}


# =========================
# 4) 冷启动“热门榜”打分与Top-K推荐
# =========================
def popularity_score(movie_id: int, douban_meta: Dict[int, Tuple[float, float]]) -> float:
    """
    score = W_RATING * douban_rating + W_VOTES * log(votes+1)
    缺失信息则给很小分
    """
    meta = douban_meta.get(movie_id)
    if meta is None:
        return -1e9
    rating, votes = meta
    if not np.isfinite(rating):
        return -1e9
    votes = max(float(votes), 0.0)
    return W_RATING * float(rating) + W_VOTES * math.log(votes + 1.0)


def build_global_popularity_ranking(
    all_item_idx: List[int],
    itemidx_to_movieid: Dict[int, int],
    douban_meta: Dict[int, Tuple[float, float]],
) -> List[int]:
    """
    生成全局热门榜（item_idx 列表），按 popularity_score 降序
    """
    scores = []
    for it in all_item_idx:
        mid = itemidx_to_movieid.get(int(it))
        if mid is None:
            s = -1e9
        else:
            s = popularity_score(mid, douban_meta)
        scores.append(s)

    ranked = [it for it, _ in sorted(zip(all_item_idx, scores), key=lambda x: x[1], reverse=True)]
    return ranked


def topk_from_global_ranking(
    global_ranked_items: List[int],
    k: int,
    exclude: set | None = None
) -> List[int]:
    """
    从全局榜单取 Top-K，可选过滤 exclude（如用户已看过的训练集物品）
    """
    if not exclude:
        return global_ranked_items[:k]
    out = []
    for it in global_ranked_items:
        if it in exclude:
            continue
        out.append(it)
        if len(out) >= k:
            break
    return out


# =========================
# 5) 主评估流程
# =========================
def evaluate_cold_start():
    rng = np.random.default_rng(RANDOM_SEED)

    df_train = pd.read_csv(TRAIN_CSV)
    df_test = pd.read_csv(TEST_CSV)

    # 必要列检查
    for col in ["user_idx", "item_idx", "movieId"]:
        if col not in df_train.columns or col not in df_test.columns:
            raise ValueError(f"train/test must contain column '{col}'")
    if "rating" not in df_test.columns:
        raise ValueError("test must contain column 'rating'")

    itemidx_to_movieid = build_itemidx_to_movieid(df_train, df_test)
    douban_meta = build_douban_meta_map(DOUBAN_INFO_CSV)

    all_items = sorted(set(df_train["item_idx"]).union(set(df_test["item_idx"])))
    global_ranked_items = build_global_popularity_ranking(all_items, itemidx_to_movieid, douban_meta)

    user_train_counts = build_user_train_counts(df_train)
    user_train_hist = build_user_train_hist(df_train)

    # 每用户一个正样本（从 test 选）
    user_pos = choose_one_positive_per_user(df_test, CHOOSE_POSITIVE, rng)
    all_users = list(user_pos.keys())

    # 分组：冷启动 vs 非冷启动
    cold_users = [u for u in all_users if user_train_counts.get(u, 0) <= COLD_THRESHOLD]
    warm_users = [u for u in all_users if user_train_counts.get(u, 0) > COLD_THRESHOLD]

    print("\n=== Cold-start Specialized Evaluation ===")
    print(f"Total users with test positive: {len(all_users)}")
    print(f"Cold-start definition: train_interactions <= {COLD_THRESHOLD}")
    print(f"Cold users: {len(cold_users)} | Warm users: {len(warm_users)}")
    print(f"Filter train interactions in recommendation: {FILTER_TRAIN_INTERACTIONS}")
    print(f"Positive selection from test: {CHOOSE_POSITIVE}")
    print("---------------------------------------------------------")

    def eval_group(users: List[int], group_name: str):
        if not users:
            print(f"[{group_name}] No users. Skipped.")
            return

        hr_sum = {k: 0.0 for k in K_LIST}
        ndcg_sum = {k: 0.0 for k in K_LIST}

        for u in users:
            pos_item = user_pos[u]
            exclude = user_train_hist.get(u, set()) if FILTER_TRAIN_INTERACTIONS else None

            # 冷启动策略：直接用全局热门榜取 Top-K（可过滤已看过）
            # 注意：这里评估是“推荐列表”，不需要负采样
            # 因为冷启动策略本质是全局榜，和负采样候选集不是同一类评估。
            for k in K_LIST:
                recs = topk_from_global_ranking(global_ranked_items, k, exclude=exclude)
                hr_sum[k] += hit_rate_at_k(recs, pos_item, k)
                ndcg_sum[k] += ndcg_at_k(recs, pos_item, k)

        n = len(users)
        print(f"\n[{group_name}] users={n}")
        for k in sorted(K_LIST):
            print(f"K={k:>2d} | HR@{k}={hr_sum[k]/n:.4f} | NDCG@{k}={ndcg_sum[k]/n:.4f}")

    # 重点：冷启动用户结果
    eval_group(cold_users, "COLD-START (Popularity only)")
    # 对比：非冷启动用户在同样热门策略下的结果（体现冷启动更难）
    eval_group(warm_users, "WARM (Popularity only)")

    print("\nNote:")
    print("- This script evaluates the cold-start fallback strategy (global popularity) as a recommendation list.")
    print("- It is NOT the same as Leave-One-Out + 99 negatives candidate evaluation used for personalized ranking models.")


if __name__ == "__main__":
    evaluate_cold_start()
