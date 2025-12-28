import requests
import re
import time
import pandas as pd
from typing import Optional, Dict, Any, List
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs  # 解析 link2 里的真实 URL

# ==========================================================
#                 配置区域（请自行修改）
# ==========================================================

# MovieLens 1M 数据路径（改成你的实际路径）
MOVIELENS_MOVIES_PATH = r".\ml-1m\movies.dat"
MOVIELENS_RATINGS_PATH = r".\ml-1m\ratings.dat"

# 爬多少部热门电影（按评分次数排序）
TOP_N = 500   # 可以先设小一点测试，之后改大

# 请求间隔
SLEEP_SECONDS_DOUBAN_SEARCH = 2.0   # 搜索页间隔
SLEEP_SECONDS_DOUBAN_DETAIL = 2.0   # 详情页间隔

# 输出文件
OUTPUT_CSV = "movie_douban_info.csv"

# 通用请求头（伪装浏览器）
HEADERS_DOUBAN = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Referer": "https://movie.douban.com/",
}


# ==========================================================
#          1. 读取 MovieLens 的 movies.dat / ratings.dat
# ==========================================================

def load_movielens_movies(path: str) -> pd.DataFrame:
    """
    MovieLens 1M movies.dat 格式：
      MovieID::Title::Genres
    例如：
      1::Toy Story (1995)::Animation|Children's|Comedy
    """
    movies: List[Dict[str, Any]] = []
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("::")
            if len(parts) < 3:
                continue
            movie_id, title_raw, genres_raw = parts

            # 从标题中提取年份
            m = re.search(r"\((\d{4})\)$", title_raw)
            if m:
                year = m.group(1)
                title = title_raw[:m.start()].strip()
            else:
                year = None
                title = title_raw.strip()

            movies.append({
                "movieId": int(movie_id),
                "ml_title": title,
                "ml_year": year,
                "ml_genres": genres_raw.split("|") if genres_raw else []
            })

    return pd.DataFrame(movies)


def load_movielens_ratings(path: str) -> pd.DataFrame:
    """
    ratings.dat 格式：
      UserID::MovieID::Rating::Timestamp
    """
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        header=None,
        names=["userId", "movieId", "rating", "timestamp"],
        encoding="latin-1"
    )
    return df


# ==========================================================
#        2. 豆瓣搜索（HTML）→ 找到 movie.douban.com 链接
# ==========================================================

def douban_search_by_html(query: str) -> Optional[Dict[str, Any]]:
    """
    使用豆瓣网页版搜索页：
      https://www.douban.com/search?cat=1002&q=xxx
    从 HTML 里提取第一个电影结果的 subject id 和实际 movie.douban.com 链接。
    只用 HTML，不用 subject_suggest，也不用 TMDB。
    """
    search_url = "https://www.douban.com/search"
    params = {
        "q": query,
        "cat": "1002"   # 1002 = 电影
    }

    try:
        resp = requests.get(search_url, headers=HEADERS_DOUBAN, params=params, timeout=10)
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        print("  [douban search page error]", e)
        return None

    soup = BeautifulSoup(html, "lxml")

    # 搜索结果外层：div.search-result > div.result-list > div.result
    search_result = soup.find("div", class_="search-result")
    if not search_result:
        return None
    result_list = search_result.find("div", class_="result-list")
    if not result_list:
        return None
    first_result = result_list.find("div", class_="result")
    if not first_result:
        return None

    # 优先从左侧图片区域的 a.nbg 拿跳转链接（link2）
    pic_div = first_result.find("div", class_="pic")
    link = None
    if pic_div:
        link = pic_div.find("a", class_="nbg")
    if not link:
        # 兜底：从标题区域拿
        content_div = first_result.find("div", class_="content")
        if content_div:
            title_div = content_div.find("div", class_="title")
            if title_div:
                link = title_div.find("a")

    if not link or not link.has_attr("href"):
        return None

    href = link["href"]

    # href 形如：https://www.douban.com/link2/?url=ENCODED_URL&cat_id=1002...
    # 需要从 query 参数里取出真正的 movie.douban.com/subject/xxxx/ 链接
    parsed = urlparse(href)
    qs = parse_qs(parsed.query)
    real_urls = qs.get("url")
    if real_urls:
        real_url = real_urls[0]
    else:
        # 如果没有 url 参数，就直接用 href 兜底
        real_url = href

    # 从真实链接里提取 subject_id
    m = re.search(r"/subject/(\d+)/", real_url)
    if not m:
        return None

    subject_id = m.group(1)
    subject_title = link.get_text(strip=True)

    return {
        "id": subject_id,
        "title": subject_title,
        "url": real_url
    }


# ==========================================================
#                3. 豆瓣详情页解析（精简字段）
# ==========================================================

def fetch_html(url: str) -> Optional[str]:
    try:
        resp = requests.get(url, headers=HEADERS_DOUBAN, timeout=10)
        resp.raise_for_status()
        resp.encoding = "utf-8"
        return resp.text
    except Exception as e:
        print("  [fetch error]", e)
        return None


def parse_douban_detail(html: str, url: str = "") -> Dict[str, Any]:
    """
    精简解析豆瓣电影详情页，只保留与推荐最直接相关的：
      - douban_genres: 类型
      - douban_country: 制片国家/地区
      - douban_rating: 豆瓣评分
      - douban_votes: 评价人数
      - douban_url: 页面链接
    """
    soup = BeautifulSoup(html, "lxml")

    data: Dict[str, Any] = {
        "douban_url": url,
        "douban_genres": None,
        "douban_country": None,
        "douban_rating": None,
        "douban_votes": None,
    }

    info_div = soup.find("div", id="info")

    # 类型：多个 <span property="v:genre">
    if info_div:
        genre_spans = info_div.find_all("span", attrs={"property": "v:genre"})
        if genre_spans:
            data["douban_genres"] = " / ".join(
                g.get_text(strip=True) for g in genre_spans
            )

    # 国家/地区（制片国家/地区）
    def extract_line(label: str) -> Optional[str]:
        if not info_div:
            return None
        span = info_div.find("span", class_="pl", string=re.compile("^" + label))
        if not span:
            return None
        parts = []
        for s in span.next_siblings:
            if getattr(s, "name", None) == "br":
                break
            txt = s.get_text(" / ", strip=True) if hasattr(s, "get_text") else s.strip()
            if txt:
                parts.append(txt)
        return "".join(parts).strip() if parts else None

    data["douban_country"] = extract_line("制片国家/地区")

    # 豆瓣评分
    rating_tag = soup.find("strong", attrs={"property": "v:average"})
    if rating_tag:
        txt = rating_tag.get_text(strip=True)
        try:
            data["douban_rating"] = float(txt)
        except ValueError:
            pass

    # 评价人数
    votes_tag = soup.find("span", attrs={"property": "v:votes"})
    if votes_tag:
        txt = votes_tag.get_text(strip=True).replace(",", "")
        try:
            data["douban_votes"] = int(txt)
        except ValueError:
            pass

    return data


# ==========================================================
#                    4. 主流程：ML → 豆瓣
# ==========================================================

def main():
    print("加载 MovieLens movies.dat & ratings.dat ...")
    movies_ml = load_movielens_movies(MOVIELENS_MOVIES_PATH)
    ratings = load_movielens_ratings(MOVIELENS_RATINGS_PATH)

    print("统计电影评分次数，选出热门 TOP_N ...")
    movie_pop = ratings.groupby("movieId").size().reset_index(name="count")
    movie_pop = movie_pop.sort_values("count", ascending=False)
    hot_ids = movie_pop["movieId"].head(TOP_N).tolist()

    movies_sub = movies_ml[movies_ml["movieId"].isin(hot_ids)].copy()
    print(f"准备处理 {len(movies_sub)} 部热门电影。")

    records: List[Dict[str, Any]] = []

    for _, row in movies_sub.iterrows():
        movie_id = row["movieId"]
        ml_title = row["ml_title"]
        ml_year = row["ml_year"]

        print(f"\n=== MovieLens: {ml_title} ({ml_year}) id={movie_id} ===")

        # ---------- 步骤 1：直接用 MovieLens 电影名搜索豆瓣 ----------
        # 可以尝试几种 query 组合：原始标题 / 带年份的标题
        queries = [ml_title]
        if ml_year:
            queries.append(f"{ml_title} ({ml_year})")

        douban_subject = None
        for q in queries:
            print(f"  尝试豆瓣搜索：{q}")
            douban_subject = douban_search_by_html(q)
            if douban_subject:
                break
            time.sleep(SLEEP_SECONDS_DOUBAN_SEARCH)

        if not douban_subject:
            print("  [warn] 豆瓣 HTML 搜索无结果。")
            continue

        douban_id = douban_subject.get("id")
        douban_url = douban_subject.get("url") or f"https://movie.douban.com/subject/{douban_id}/"
        print(f"  豆瓣命中: id={douban_id}, url={douban_url}")

        time.sleep(SLEEP_SECONDS_DOUBAN_DETAIL)

        # ---------- 步骤 2：爬豆瓣详情 ----------
        html = fetch_html(douban_url)
        if not html:
            print("  [warn] 豆瓣详情页获取失败。")
            continue

        douban_detail = parse_douban_detail(html, url=douban_url)

        # ---------- 汇总记录 ----------
        record: Dict[str, Any] = {
            "movieId": movie_id,
            "ml_title": ml_title,
            "ml_year": ml_year,
            "ml_genres": "|".join(row["ml_genres"]),
        }
        record.update(douban_detail)

        records.append(record)

        # 详情页之间暂停一下，避免太频繁
        time.sleep(SLEEP_SECONDS_DOUBAN_DETAIL)

    # ---------- 保存结果 ----------
    df_out = pd.DataFrame(records)
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    print(f"\n处理完成，共成功获取 {len(df_out)} 部电影的豆瓣信息。")
    print(f"结果已保存至：{OUTPUT_CSV}")


if __name__ == "__main__":
    main()
