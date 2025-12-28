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
#   MCP 工具 1：单个评分预测（返回 JSON 字符串）
# ======================

@mcp.tool()
def predict_rating(user_idx: int, item_idx: int) -> str:
    """
    预测用户对某电影的偏好评分。
    返回值是 JSON 字符串，方便 Dify 处理。
    """
    # 简单边界检查
    if item_idx < 0 or item_idx >= num_items:
        return json.dumps({
            "error": "item_idx out of range",
            "user_idx": user_idx,
            "item_idx": item_idx
        }, ensure_ascii=False)

    u = np.array([user_idx], dtype="int32")
    i = np.array([item_idx], dtype="int32")
    feat = X_item_all[item_idx:item_idx+1]

    pred = model.predict([u, i, feat], verbose=0)[0, 0]

    result = {
        "user_idx": int(user_idx),
        "item_idx": int(item_idx),
        "pred_rating": float(pred)
    }
    return "666" + json.dumps(result, ensure_ascii=False)


# ======================
#   MCP 工具 2：Top-K 推荐（返回 JSON 字符串）
# ======================

@mcp.tool()
def recommend(user_idx: int, top_k: int = 10) -> str:
    """
    返回用户的 Top-K 推荐结果。
    返回值为 JSON 字符串，结构形如：
    {
      "user_idx": ...,
      "top_k": ...,
      "items": [
        {
          "item_idx": ...,
          "movieId": ...,
          "score": ...,
          "title": "...",
          "genres": "...",
          "douban_rating": ...,
          "douban_url": "..."
        },
        ...
      ]
    }
    """
    # 构造批量输入：对于该 user，对所有 item 进行预测
    user_arr = np.full((num_items,), user_idx, dtype="int32")
    item_arr = np.arange(num_items, dtype="int32")

    preds = model.predict([user_arr, item_arr, X_item_all], verbose=0).reshape(-1)

    # 取 Top-K item_idx
    top_k = max(1, min(top_k, num_items))  # 防止越界或非法
    top_idxs = preds.argsort()[::-1][:top_k]

    items: List[dict] = []
    for idx in top_idxs:
        movie_id = idx2movie.get(int(idx))
        meta = movie_meta.get(movie_id, {}) if movie_id is not None else {}

        # movieId 可能不存在，做个保护
        movie_id_safe = int(movie_id) if movie_id is not None and not pd.isna(movie_id) else None

        items.append({
            "item_idx": int(idx),
            "movieId": movie_id_safe,
            "score": float(preds[idx]),
            "title": meta.get("ml_title"),
            "genres": meta.get("ml_genres"),
            "douban_rating": meta.get("douban_rating"),
            "douban_url": meta.get("douban_url"),
        })

    result = {
        "user_idx": int(user_idx),
        "top_k": int(top_k),
        "items": items
    }
    return "666" + json.dumps(result, ensure_ascii=False)

@mcp.tool()
def recommend2(user_idx: int, top_k: int = 10) -> str:
    """
    两阶段推荐：
    1）召回（Recall）：基于豆瓣评分 + 热度的 Top-N
    2）排序（Rank）：NeuMF 对候选集精排
    返回 JSON 字符串（符合 Dify 要求）
    """

    # ========== 1️⃣ 召回阶段（Recall） ==========
    # 召回池大小，可调：100 / 200 / 300
    RECALL_SIZE = 200

    # 使用豆瓣评分 & 评价人数进行召回
    recall_df = douban_df.copy()

    # 防止缺失值
    recall_df["douban_rating"] = recall_df["douban_rating"].fillna(0)
    recall_df["douban_votes"] = recall_df["douban_votes"].fillna(0)

    # 简单加权排序（你也可以只按 rating）
    recall_df["recall_score"] = (
        recall_df["douban_rating"] * 0.7 +
        np.log1p(recall_df["douban_votes"]) * 0.3
    )

    recall_df = recall_df.sort_values(
        by="recall_score", ascending=False
    ).head(RECALL_SIZE)

    # movieId → item_idx
    recall_movie_ids = recall_df["movieId"].tolist()

    recall_item_idxs = [
        int(itemid_map[itemid_map["movieId"] == mid]["item_idx"].values[0])
        for mid in recall_movie_ids
        if mid in idx2movie.values()
    ]

    if len(recall_item_idxs) == 0:
        return json.dumps({
            "error": "Recall stage returned empty candidate set"
        }, ensure_ascii=False)

    # ========== 2️⃣ 排序阶段（NeuMF Rank） ==========
    user_arr = np.full(len(recall_item_idxs), user_idx, dtype="int32")
    item_arr = np.array(recall_item_idxs, dtype="int32")
    feat_arr = X_item_all[item_arr]

    preds = model.predict(
        [user_arr, item_arr, feat_arr],
        verbose=0
    ).reshape(-1)

    # 排序取 Top-K
    top_k = min(top_k, len(preds))
    top_indices = preds.argsort()[::-1][:top_k]

    # ========== 3️⃣ 组装结果 ==========
    results = []
    for rank_idx in top_indices:
        item_idx = int(item_arr[rank_idx])
        movie_id = idx2movie.get(item_idx)
        meta = movie_meta.get(movie_id, {})

        results.append({
            "item_idx": item_idx,
            "movieId": int(movie_id) if movie_id is not None else None,
            "score": float(preds[rank_idx]),
            "title": meta.get("ml_title"),
            "genres": meta.get("ml_genres"),
            "douban_rating": meta.get("douban_rating"),
            "douban_url": meta.get("douban_url"),
        })

    return json.dumps({
        "user_idx": int(user_idx),
        "recall_size": len(recall_item_idxs),
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
