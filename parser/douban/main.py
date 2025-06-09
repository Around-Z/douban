# scraper_douban.py

import requests
import lxml.html
import time
import os
import re
from requests.exceptions import RequestException
import random
import traceback

# --- 导入自定义模块 ---
from database_douban import get_db_connection, store_douban_movie
from parser_douban import get_basic_data, get_douban_detail_data

# --- 全局配置与请求头 ---
DOUBAN_TOP250_URL = "https://movie.douban.com/top250?start=0&filter="
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
    'Cookie': '__utmv=30149280.26588; _vwo_uuid_v2=DAC63BADB41673F0603FA177C6F5344C6|23d9f2799e7e65cc292a49365de88a9c; __utma=30149280.2038156265.1708787605.1721631000.1727963319.9; __utma=223695111.1132091569.1709467909.1715697793.1727963319.4; bid=i3lhJgsUsJk; ll="118174"; dbcl2="265884385:DBbxLcXmPq0"; push_noty_num=0; push_doumail_num=0; ck=ATer; frodotk_db="54011ed41ddd306dfc035c7c59979147"; ap_v=0,6.0',
    'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
    'Referer': 'https://movie.douban.com/'
}


def fetch_html(url: str, retries: int = 3) -> str:
    """一个健壮的HTML获取函数"""
    delay = random.uniform(0.5, 1.2)
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


def crawl_top250_task(db_connection):
    """执行爬取豆瓣Top250的任务"""
    print(f"\n{'=' * 20} 开始 [豆瓣Top250] 爬取任务 {'=' * 20}")
    for i in range(10):
        start_index = i * 25
        page_num = i + 1
        print(f"\n--- 正在处理第 {page_num}/10 页 ---")
        list_url = DOUBAN_TOP250_URL.replace('start=0', f'start={start_index}')
        list_html = fetch_html(list_url)
        if not list_html: continue

        selector = lxml.html.fromstring(list_html)
        movie_items = selector.xpath('//div[@class="item"]')
        for idx, item_selector in enumerate(movie_items):
            try:
                name1, name2, score, comment, quote, page_url, douban_id = get_basic_data(item_selector)
                print(f"🎬 正在处理 No.{start_index + idx + 1}: {name1}")
                if not page_url: continue

                detail_html = fetch_html(page_url)
                if not detail_html: continue

                detail_data = get_douban_detail_data(detail_html)

                # 合并所有数据
                full_data = {
                    "ranking": start_index + idx + 1, "douban_id": douban_id, "name_cn": name1,
                    "score": float(score), "comment_count": int(comment), "quote": quote, "douban_url": page_url,
                    **detail_data
                }
                store_douban_movie(db_connection, full_data)
            except Exception as e:
                print(f"  💥  处理电影 '{name1}' 时发生未知错误: {e}")
                traceback.print_exc()


# --- 主程序入口 ---
if __name__ == "__main__":
    db_conn = None
    try:
        db_conn = get_db_connection()
        if not db_conn: exit()

        print("\n" + "=" * 50)
        print("     豆瓣Top250电影数据采集程序已准备就绪")
        print("=" * 50)
        confirm = input("本操作将开始爬取数据并覆盖数据库中的旧记录。是否继续？(yes/no): ")
        if confirm.lower() != 'yes':
            print("操作已取消。")
            exit()

        crawl_top250_task(db_conn)

    except Exception as e:
        print(f"❌  主程序发生严重错误: {e}\n{traceback.format_exc()}")
    finally:
        if db_conn:
            db_conn.close()
            print("\n✔️  所有任务完成，数据库连接已关闭。")