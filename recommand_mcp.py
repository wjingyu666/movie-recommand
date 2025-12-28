from fastmcp import FastMCP
from typing import List
import numpy as np
import pandas as pd
import tensorflow as tf
import json

# ======================
#   初始化 MCP 服务
# ======================
mcp = FastMCP("NeuMFRecommenderMCP")

# ======================
#   加载训练好的模型
# ======================

print("Loading NeuMF model ...")
model = tf.keras.models.load_model("neumf_model.h5", compile=False)

print("Loading metadata ...")
train_df = pd.read_csv("train_data.csv")          # 用于 item_idx -> 特征向量
douban_df = pd.read_csv("movie_douban_info.csv")  # 电影元信息：标题、类型、豆瓣评分等

# 物品特征列
feature_cols = [c for c in train_df.columns if c.startswith("feat_")]
print("Item feature columns:", feature_cols)

# item_idx -> movieId
itemid_map = train_df[["movieId", "item_idx"]].drop_duplicates()
idx2movie = dict(zip(itemid_map["item_idx"], itemid_map["movieId"]))

# movieId -> 电影元数据（字典）
movie_meta = douban_df.set_index("movieId").to_dict(orient="index")

# 构造 item_idx -> 特征向量 矩阵
item_feat_table = (
    train_df[["item_idx"] + feature_cols]
    .drop_duplicates(subset="item_idx")
    .set_index("item_idx")
    .sort_index()
)
X_item_all = item_feat_table.values.astype("float32")

num_items = X_item_all.shape[0]
num_features = X_item_all.shape[1]
print("num_items =", num_items, "num_features =", num_features)

# ======================
#   冷启动：识别“已知用户”
# ======================
# 训练集里出现过的 user_idx
if "user_idx" in train_df.columns:
    known_user_set = set(train_df["user_idx"].unique().tolist())
    num_users = int(train_df["user_idx"].max() + 1)
else:
    # 万一你的 train_data.csv 没保存 user_idx（一般不会）
    known_user_set = set()
    num_users = None

print("known users:", len(known_user_set), "num_users:", num_users)


# ======================
#   冷启动兜底推荐：只基于内容/热度（不走模型）
# ======================
def cold_start_recommend(top_k: int = 10, recall_size: int = 200) -> List[dict]:
    """
    冷启动策略：直接返回豆瓣“高分 + 高热度”的 Top-K
    （相当于只做召回，不做个性化排序）
    """
    df = douban_df.copy()
    df["douban_rating"] = df.get("douban_rating", 0).fillna(0)
    df["douban_votes"] = df.get("douban_votes", 0).fillna(0)

    df["recall_score"] = df["douban_rating"] * 0.7 + np.log1p(df["douban_votes"]) * 0.3
    df = df.sort_values(by="recall_score", ascending=False).head(recall_size)

    # movieId → item_idx（只保留能映射到 item_idx 的电影）
    recall_movie_ids = df["movieId"].tolist()
    recall_item_idxs = []
    for mid in recall_movie_ids:
        rows = itemid_map[itemid_map["movieId"] == mid]
        if len(rows) > 0:
            recall_item_idxs.append(int(rows["item_idx"].values[0]))

    # 直接取前 top_k（不走 NeuMF）
    recall_item_idxs = recall_item_idxs[:max(1, min(top_k, len(recall_item_idxs)))]

    results = []
    for item_idx in recall_item_idxs:
        movie_id = idx2movie.get(item_idx)
        meta = movie_meta.get(movie_id, {})
        results.append({
            "item_idx": int(item_idx),
            "movieId": int(movie_id) if movie_id is not None else None,
            "score": None,  # 冷启动无模型分数
            "title": meta.get("ml_title"),
            "genres": meta.get("ml_genres"),
            "douban_rating": meta.get("douban_rating"),
            "douban_url": meta.get("douban_url"),
            "reason": "cold_start_popular"  # 方便你在报告/前端解释
        })
    return results


# ======================
#   MCP 工具 1：单个评分预测（返回 JSON 字符串）
# ======================

@mcp.tool()
def predict_rating(user_idx: int, item_idx: int) -> str:
    """
    预测用户对某电影的偏好评分（仅对已知用户有效）。
    返回 JSON 字符串，方便 Dify 处理。
    """
    # item 边界检查
    if item_idx < 0 or item_idx >= num_items:
        return json.dumps({
            "error": "item_idx out of range",
            "user_idx": int(user_idx),
            "item_idx": int(item_idx)
        }, ensure_ascii=False)

    # 冷启动用户：不支持评分预测（没有 user embedding）
    if user_idx not in known_user_set:
        return json.dumps({
            "error": "cold_start_user_not_supported_for_predict",
            "user_idx": int(user_idx),
            "item_idx": int(item_idx),
            "hint": "Use recommend() without known user_idx, or implement user profile based cold-start."
        }, ensure_ascii=False)

    u = np.array([user_idx], dtype="int32")
    i = np.array([item_idx], dtype="int32")
    feat = X_item_all[item_idx:item_idx+1]

    pred = model.predict([u, i, feat], verbose=0)[0, 0]

    return json.dumps({
        "user_idx": int(user_idx),
        "item_idx": int(item_idx),
        "pred_rating": float(pred)
    }, ensure_ascii=False)


# ======================
#   MCP 工具 2：Top-K 推荐（返回 JSON 字符串）
# ======================

@mcp.tool()
def recommend(user_idx: int, top_k: int = 10) -> str:
    """
    两阶段推荐 + 冷启动：
    - 已知用户：Recall（豆瓣Top-N）→ NeuMF 排序 → Top-K
    - 新用户：直接走冷启动兜底（豆瓣高分+热度 Top-K）
    返回 JSON 字符串（符合 Dify 要求）
    """

    top_k = int(max(1, top_k))

    # ========== 0️⃣ 冷启动判断 ==========
    is_cold_start = (user_idx not in known_user_set)

    # ========== 冷启动分支：不走模型 ==========
    if is_cold_start:
        items = cold_start_recommend(top_k=top_k, recall_size=200)
        return json.dumps({
            "user_idx": int(user_idx),
            "cold_start": True,
            "strategy": "popular_by_douban_rating_votes",
            "top_k": int(top_k),
            "items": items
        }, ensure_ascii=False)

    # ========== 1️⃣ 召回阶段（Recall） ==========
    RECALL_SIZE = 200

    recall_df = douban_df.copy()
    recall_df["douban_rating"] = recall_df.get("douban_rating", 0).fillna(0)
    recall_df["douban_votes"] = recall_df.get("douban_votes", 0).fillna(0)

    recall_df["recall_score"] = (
        recall_df["douban_rating"] * 0.7 +
        np.log1p(recall_df["douban_votes"]) * 0.3
    )

    recall_df = recall_df.sort_values(by="recall_score", ascending=False).head(RECALL_SIZE)

    # movieId → item_idx
    recall_movie_ids = recall_df["movieId"].tolist()
    recall_item_idxs = []
    for mid in recall_movie_ids:
        rows = itemid_map[itemid_map["movieId"] == mid]
        if len(rows) > 0:
            recall_item_idxs.append(int(rows["item_idx"].values[0]))

    if len(recall_item_idxs) == 0:
        return json.dumps({
            "error": "Recall stage returned empty candidate set",
            "user_idx": int(user_idx)
        }, ensure_ascii=False)

    # ========== 2️⃣ 排序阶段（NeuMF Rank） ==========
    user_arr = np.full(len(recall_item_idxs), user_idx, dtype="int32")
    item_arr = np.array(recall_item_idxs, dtype="int32")
    feat_arr = X_item_all[item_arr]

    preds = model.predict([user_arr, item_arr, feat_arr], verbose=0).reshape(-1)

    # Top-K
    top_k = min(top_k, len(preds))
    top_indices = preds.argsort()[::-1][:top_k]

    # ========== 3️⃣ 组装结果 ==========
    results = []
    for rank_pos in top_indices:
        item_idx = int(item_arr[rank_pos])
        movie_id = idx2movie.get(item_idx)
        meta = movie_meta.get(movie_id, {})

        results.append({
            "item_idx": item_idx,
            "movieId": int(movie_id) if movie_id is not None else None,
            "score": float(preds[rank_pos]),
            "title": meta.get("ml_title"),
            "genres": meta.get("ml_genres"),
            "douban_rating": meta.get("douban_rating"),
            "douban_url": meta.get("douban_url"),
            "reason": "neumf_rank_on_recall_candidates"
        })

    return json.dumps({
        "user_idx": int(user_idx),
        "cold_start": False,
        "recall_size": int(len(recall_item_idxs)),
        "top_k": int(top_k),
        "items": results
    }, ensure_ascii=False)


# ======================
#   启动 MCP 服务
# ======================

if __name__ == "__main__":
    print("NeuMF MCP server running on http://0.0.0.0:9000/mcp ...")
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=9000,
        path="/mcp"
    )