import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from collections import Counter

# ==========================================================
#                 路径配置（请按你的实际情况修改）
# ==========================================================

MOVIELENS_RATINGS_PATH = r"./ml-1m/ratings.dat"
MOVIELENS_MOVIES_PATH  = r"./ml-1m/movies.dat"
DOUBAN_INFO_PATH       = r"movie_douban_info.csv"  # 前面爬虫生成的文件

OUTPUT_TRAIN_PATH      = r"train_data.csv"
OUTPUT_TEST_PATH       = r"test_data.csv"

# 特征工程参数
TOP_N_GENRES   = 20   # 选最常见的前 N 个类型做 one-hot
TOP_N_COUNTRY  = 10   # 选最常见的前 N 个国家做 one-hot
TEST_SIZE      = 0.2  # 测试集比例
RANDOM_STATE   = 42


# ==========================================================
#                 一些辅助函数
# ==========================================================

def parse_ml_genres(s):
    """解析 MovieLens 的类型字段：'Action|Drama|Sci-Fi' → ['Action','Drama','Sci-Fi']"""
    if pd.isna(s):
        return []
    return [g.strip() for g in str(s).split('|') if g and g.lower() != '(no genres listed)']


def parse_douban_genres(s):
    """解析豆瓣类型字段：'剧情 / 犯罪' → ['剧情','犯罪']"""
    if pd.isna(s):
        return []
    parts = str(s).replace('／', '/').split('/')
    return [p.strip() for p in parts if p.strip()]


def parse_country(s):
    """豆瓣制片国家/地区：'美国 / 英国' → 主国家 '美国'"""
    if pd.isna(s):
        return None
    parts = str(s).replace('／', '/').split('/')
    main = parts[0].strip()
    return main if main else None


# ==========================================================
#                 1. 读取原始数据
# ==========================================================

print("加载 ratings.dat ...")
ratings = pd.read_csv(
    MOVIELENS_RATINGS_PATH,
    sep="::",
    engine="python",
    header=None,
    names=["userId", "movieId", "rating", "timestamp"],
    encoding="latin-1"
)

print("加载 movies.dat ...")
movies_raw = pd.read_csv(
    MOVIELENS_MOVIES_PATH,
    sep="::",
    engine="python",
    header=None,
    names=["movieId", "raw_title", "ml_genres"],
    encoding="latin-1"
)

print("加载 movie_douban_info.csv ...")
douban = pd.read_csv(DOUBAN_INFO_PATH, encoding="utf-8")


# ==========================================================
#       2. 处理 MovieLens 电影信息（提取年份 + 纯标题）
# ==========================================================

title_year_pattern = r"\((\d{4})\)\s*$"

movies_raw["ml_year"] = movies_raw["raw_title"].str.extract(title_year_pattern, expand=False)
movies_raw["ml_year"] = movies_raw["ml_year"].astype("float").astype("Int64")  # 可能有缺失

movies_raw["ml_title"] = movies_raw["raw_title"].str.replace(title_year_pattern, "", regex=True).str.strip()

movies_ml = movies_raw[["movieId", "ml_title", "ml_year", "ml_genres"]]


# ==========================================================
#       3. 合并 MovieLens 电影信息 和 豆瓣信息
# ==========================================================

print("合并 MovieLens 电影信息 与 豆瓣信息 ...")
movies_merged = pd.merge(
    movies_ml,
    douban,
    on="movieId",
    how="inner",
    suffixes=("_ml", "_db")
)

print(f"合并后电影数：{len(movies_merged)}")
# ----★ 关键修正：统一 ml_genres 列名 ----
# 合并后可能有：ml_genres_ml、ml_genres_db，或者只有一个
if "ml_genres_ml" in movies_merged.columns:
    movies_merged["ml_genres"] = movies_merged["ml_genres_ml"]
elif "ml_genres" not in movies_merged.columns and "ml_genres_db" in movies_merged.columns:
    movies_merged["ml_genres"] = movies_merged["ml_genres_db"]
# 也可以顺手统一标题和年份（可选）
if "ml_title_ml" in movies_merged.columns:
    movies_merged["ml_title"] = movies_merged["ml_title_ml"]
if "ml_year_ml" in movies_merged.columns:
    movies_merged["ml_year"] = movies_merged["ml_year_ml"]


# ==========================================================
#       4. 将评分数据和电影信息合并（只保留有豆瓣数据的 movieId）
# ==========================================================

print("将评分数据与电影侧信息合并 ...")
data = pd.merge(
    ratings,
    movies_merged,
    on="movieId",
    how="inner"
)

print(f"合并后评分条数：{len(data)}")


# ==========================================================
#       5. 编码 userId, movieId → user_idx, item_idx
# ==========================================================

print("LabelEncoder 编码 userId 和 movieId ...")
user_encoder = LabelEncoder()
item_encoder = LabelEncoder()

data["user_idx"] = user_encoder.fit_transform(data["userId"])
data["item_idx"] = item_encoder.fit_transform(data["movieId"])

num_users = data["user_idx"].nunique()
num_items = data["item_idx"].nunique()
print(f"用户数（num_users）: {num_users}")
print(f"物品数（num_items）: {num_items}")


# ==========================================================
#       6. 电影侧特征工程（在去重后的 movie 表上做）
# ==========================================================

print("构建电影侧特征 ...")
items = movies_merged.drop_duplicates(subset="movieId").copy()

# ---------- 6.1 类型（genres）多标签 one-hot ----------
items["ml_genres_list"] = items["ml_genres"].apply(parse_ml_genres)
items["douban_genres_list"] = items.get("douban_genres", pd.Series([None]*len(items))).apply(parse_douban_genres)

items["all_genres_list"] = items.apply(
    lambda row: list(set(row["ml_genres_list"] + row["douban_genres_list"])),
    axis=1
)

genre_counter = Counter()
for genres in items["all_genres_list"]:
    genre_counter.update(genres)

top_genres = [g for g, _ in genre_counter.most_common(TOP_N_GENRES)]
print("选取的 TOP genres:", top_genres)

for g in top_genres:
    col_name = f"feat_genre_{g}"
    items[col_name] = items["all_genres_list"].apply(lambda lst: 1 if g in lst else 0)

# ---------- 6.2 国家 one-hot + Other ----------
items["main_country"] = items.get("douban_country", pd.Series([None]*len(items))).apply(parse_country)
country_counter = Counter(c for c in items["main_country"] if pd.notna(c))
top_countries = [c for c, _ in country_counter.most_common(TOP_N_COUNTRY)]
print("选取的 TOP countries:", top_countries)

# Top-K 国家
for c in top_countries:
    col_name = f"feat_country_{c}"
    items[col_name] = (items["main_country"] == c).astype(int)

# 其他国家统一编码到 feat_country_Other
items["feat_country_Other"] = items["main_country"].apply(
    lambda c: 0 if pd.isna(c) else (0 if c in top_countries else 1)
)

# ---------- 6.3 豆瓣评分 & 评价人数 数值特征 ----------
items["douban_rating_filled"] = items.get("douban_rating", pd.Series([np.nan]*len(items))).astype(float)
items["douban_rating_filled"] = items["douban_rating_filled"].fillna(items["douban_rating_filled"].mean())

items["douban_votes_filled"] = items.get("douban_votes", pd.Series([0]*len(items))).fillna(0)
items["douban_votes_filled"] = items["douban_votes_filled"].astype(float)

items["douban_votes_log"] = np.log1p(items["douban_votes_filled"])

scaler = MinMaxScaler()
items[["feat_rating_norm", "feat_votes_norm"]] = scaler.fit_transform(
    items[["douban_rating_filled", "douban_votes_log"]]
)

item_feature_cols = [c for c in items.columns if c.startswith("feat_")]
print(f"电影侧特征维度：{len(item_feature_cols)} 个特征列")


# ==========================================================
#       7. 将电影特征合并回评分数据（按 movieId）
# ==========================================================

print("将电影特征合并回评分数据 ...")
data = pd.merge(
    data,
    items[["movieId"] + item_feature_cols],
    on="movieId",
    how="left"
)

data[item_feature_cols] = data[item_feature_cols].fillna(0.0)

print("合并后的数据列：", data.columns.tolist())


# ==========================================================
#       8. 划分训练集 / 测试集
# ==========================================================

print("划分训练集 / 测试集 ...")
train_df, test_df = train_test_split(
    data,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

print(f"训练集样本数：{len(train_df)}, 测试集样本数：{len(test_df)}")


# ==========================================================
#       9. 保存为 CSV（用于后续 NCF 训练）
# ==========================================================

base_cols = ["userId", "movieId", "user_idx", "item_idx", "rating"]
save_cols = base_cols + item_feature_cols

train_df[save_cols].to_csv(OUTPUT_TRAIN_PATH, index=False, encoding="utf-8-sig")
test_df[save_cols].to_csv(OUTPUT_TEST_PATH, index=False, encoding="utf-8-sig")

print(f"训练数据已保存到：{OUTPUT_TRAIN_PATH}")
print(f"测试数据已保存到：{OUTPUT_TEST_PATH}")
print("预处理 + 特征工程完成。")
