# maoyan_scraper.py

import requests
import lxml.html
import time
import re
from urllib.parse import quote
from requests.exceptions import RequestException
import random
import traceback

# --- 导入自定义模块 ---
from database import get_db_connection, store_maoyan_movie
from movie_parser import parse_maoyan_rankings, find_best_match_from_douban_search, get_douban_detail_data

# --- 全局配置与请求头 ---
MAOYAN_URL = 'https://piaofang.maoyan.com/rankings/year'
DOUBAN_SEARCH_URL = 'https://www.douban.com/search?cat=1002&q='
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
    'Cookie': '__utmv=30149280.26588; _vwo_uuid_v2=DAC63BADB41673F0603FA177C6F5344C6|23d9f2799e7e65cc292a49365de88a9c; __utma=30149280.2038156265.1708787605.1721631000.1727963319.9; __utma=223695111.1132091569.1709467909.1715697793.1727963319.4; bid=i3lhJgsUsJk; ll="118174"; dbcl2="265884385:DBbxLcXmPq0"; push_noty_num=0; push_doumail_num=0; ck=ATer; frodotk_db="54011ed41ddd306dfc035c7c59979147"; ap_v=0,6.0',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Referer': 'https://www.douban.com/'
}


def get_html(url: str, retries: int = 3) -> str:
    """健壮的HTML获取函数"""
    delay = random.uniform(0.5, 1)
    print(f"  🌐  正在请求: {url} (延迟 {delay:.2f}s)")
    time.sleep(delay)
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=HEADERS, timeout=30)
            response.raise_for_status()
            if len(response.text) < 1000: raise RequestException("获取到的HTML内容过短")
            return response.text
        except RequestException as e:
            print(f"  ❌  网络请求失败 (第 {attempt + 1} 次尝试): {e}")
            if attempt < retries - 1: time.sleep(5 * (attempt + 1))
    return ""


def get_search_url(text: str) -> str:
    return DOUBAN_SEARCH_URL + quote(text)


def main():
    db_conn = get_db_connection()
    if not db_conn: return

    print("--- 开始爬取猫眼票房榜单 ---")
    maoyan_html = get_html(MAOYAN_URL)
    maoyan_movies = parse_maoyan_rankings(maoyan_html)

    if not maoyan_movies:
        print("❌ 未能从猫眼票房榜获取任何电影数据，程序终止。")
        return

    print(f"✔️  成功从猫眼获取 {len(maoyan_movies)} 部电影基础数据。")

    for i, movie_base in enumerate(maoyan_movies):
        print(f"\n--- 正在处理第 {i + 1}/{len(maoyan_movies)} 部电影: {movie_base['name']} ---")
        try:
            # 1. 豆瓣搜索
            search_year = int(movie_base['release_date'].split('-')[0])
            search_query = f"({search_year}) {movie_base['name']}"
            print(f"  🔍  正在豆瓣搜索: '{search_query}'")
            search_html = get_html(get_search_url(search_query))
            if not search_html: continue

            douban_match = find_best_match_from_douban_search(search_html)

            # 2. 初始化最终数据字典
            full_data = {**movie_base, "douban_id": None, "douban_score": 0.0, "douban_comment_count": 0,
                         "cover_url": "", "synopsis": "", "comments_json": "[]", "directors": [], "actors": [],
                         "genres": [], "name_en": "", "quote": "-", "release_date_text": "", "runtime_text": "",
                         "year": search_year, "douban_url": ""}

            # 3. 深入爬取详情页
            if douban_match and douban_match.get("douban_url"):
                print(f"  ✔️  找到匹配项，正在深入详情页...")
                full_data.update(douban_match)

                detail_html = get_html(douban_match["douban_url"])
                if detail_html:
                    # get_douban_detail_data会返回包含genres的字典
                    douban_detail_data = get_douban_detail_data(detail_html)
                    # update会把genres列表更新到full_data中
                    full_data.update(douban_detail_data)
                    print(f"  ✔️  成功解析豆瓣详情页，类型: {full_data.get('genres')}")
            else:
                print(f"  ❌  在豆瓣未找到匹配项。")

            # 4. 存入数据库
            # store_maoyan_movie现在可以正确处理full_data中包含的genres列表
            store_maoyan_movie(db_conn, full_data)

        except Exception:
            print(f"  💥  处理电影 '{movie_base.get('name')}' 时发生未知严重错误:")
            traceback.print_exc()
            continue

    if db_conn:
        db_conn.close()
        print("\n✔️  所有任务完成，数据库连接已关闭。")


if __name__ == "__main__":
    main()