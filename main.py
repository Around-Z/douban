import requests
import lxml.html
import pandas as pd
import time
import os
import re
from urllib.parse import quote, unquote
from requests.exceptions import RequestException

# 导入自定义模块
from movie_detail import get_detail_data
from movie_basic import get_basic_data
from database import db_store, db_store_2, csv_store, db
from attachfile import headers

# --- 全局配置 ---
local_test = False

# --- 全局URL定义 ---
douban_top250_url = "https://movie.douban.com/top250?start=0&filter="
maoyan_rankings_url = 'https://piaofang.maoyan.com/rankings/year'
douban_search_prefix_url = 'https://www.douban.com/search?cat=1002&q='


# --- 辅助函数：HTML获取与保存 (优化了重试和延迟) ---
def _get_or_load_html(url: str, file_path: str, is_local_test: bool, retries: int = 3, delay: float = 1.0) -> str:
    if is_local_test:
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding='utf-8') as f:
                    print(f"Loading HTML from local file: {file_path}")
                    return f.read()
            except Exception as e:
                print(f"Error reading local file {file_path}: {e}")
                return ""
        else:
            print(f"Warning: Local file {file_path} not found for local_test mode. Skipping.")
            return ""
    else:
        for attempt in range(retries):
            try:
                print(f"Fetching URL: {url} (Attempt {attempt + 1}/{retries})")
                time.sleep(delay)
                response = requests.get(url, headers=headers, timeout=15)
                response.raise_for_status()
                html_content = response.content.decode('utf-8')

                os.makedirs(os.path.dirname(file_path) or './html', exist_ok=True)
                with open(file_path, 'w+', encoding='utf-8') as f:
                    f.write(html_content)
                print(f"HTML saved to {file_path}")
                time.sleep(0.5)
                return html_content
            except RequestException as e:
                print(f"Error fetching URL {url}: {e}")
                if attempt == retries - 1:
                    print(f"Failed to fetch {url} after {retries} attempts. Skipping.")
                    return ""
                time.sleep(delay * (attempt + 1))
        return ""


def get_url(base_url: str, start: int) -> str:
    return base_url.replace('start=0', 'start=' + str(start))


def get_search_url(text: str) -> str:
    """生成豆瓣搜索 URL，编码电影名称以处理特殊字符"""
    return douban_search_prefix_url + quote(text)


# --- 豆瓣电影TOP250数据爬取 (get_data) ---
def get_data(base_url: str):
    movies_data = []
    for i in range(10):
        start_index = i * 25
        page_num = i + 1
        print(f"\n--- Processing Douban Top250 Page {page_num} (start={start_index}) ---")

        _url = get_url(base_url, start=start_index)
        html_file_path = f"./html/douban_top250_page_{start_index}.html"
        current_page_html = _get_or_load_html(_url, html_file_path, local_test)

        if not current_page_html:
            print(f"Skipping Douban Top250 page {page_num} due to empty HTML.")
            continue

        selector = lxml.html.fromstring(current_page_html)
        movie_divs = selector.xpath('//div[@class="item"]')

        if not movie_divs:
            print(f"Warning: No movie items found on Douban Top250 page {page_num}.")
            continue

        for movie_idx, movie_item_selector in enumerate(movie_divs):
            movie_name_for_log = "Unknown Movie"
            try:
                name1, name2, score, comment, quote_str, page_url_detail = get_basic_data(movie_item_selector)
                movie_name_for_log = name1 if name1 and name1 != '-' else (
                    name2 if name2 and name2 != '-' else "Unknown Movie")

                print(f"  Processing Douban movie {movie_idx + 1} on page {page_num}: '{movie_name_for_log}'")

                if not page_url_detail:
                    print(f"  Warning: No detail URL found for '{movie_name_for_log}'. Skipping detail crawling.")
                    continue

                detail_html_filename = re.sub(r'[^\w\s-]', '', name1 if name1 != '-' else name2)
                detail_html_filename = detail_html_filename.strip()[:50]
                detail_html_file_path = f"./html/douban_detail_{detail_html_filename}.html"

                movie_detail_html = _get_or_load_html(page_url_detail, detail_html_file_path, local_test)

                if not movie_detail_html:
                    print(f"  Skipping detail parsing for '{movie_name_for_log}' due to empty HTML.")
                    continue

                director, actor, movie_type, place, lang, year, length = get_detail_data(movie_detail_html)

                movie_data_combined = [
                    name1, name2, score, comment, quote_str, page_url_detail,
                    director, actor, movie_type, place, lang, year, length
                ]
                db_store(movie_data_combined)
                movies_data.append(movie_data_combined)

            except ValueError as ve:
                print(f"  ValueError processing '{movie_name_for_log}' details: {ve}")
            except Exception as e:
                print(f"  Error processing movie '{movie_name_for_log}' on page {page_num}: {e}")

    csv_store()

    if movies_data:
        df_douban = pd.DataFrame(movies_data, columns=[
            '中文名', '外文名', '评分', '评价人数', '电影语录', '详情URL', '导演', '主演', '类型', '地区', '语言',
            '上映年份', '时长'
        ])
        os.makedirs("./csv", exist_ok=True)
        df_douban.to_csv("./csv/豆瓣Top250数据.csv", index=False, encoding='utf-8-sig')
        print("豆瓣Top250数据已导出到 豆瓣Top250数据.csv")
    else:
        print("No Douban Top250 data collected for CSV export.")

    return pd.DataFrame(movies_data)


# --- 豆瓣搜索结果页面解析 (get_data_douban) ---
def get_data_douban(movie_name: str, html_content: str, maoyan_release_date: str) -> tuple:
    selector = lxml.html.fromstring(html_content)

    best_match_info = {
        'score': '0.0',
        'comment': 0,
        'page_url': '',
        'match_score': -1
    }

    search_results = selector.xpath('//div[@class="result-list"]/div[@class="result"]')

    print(
        f"  DEBUG in get_data_douban: Searching for '{movie_name}' (Maoyan Release Date: {maoyan_release_date}) found {len(search_results)} results.")

    maoyan_year = int(maoyan_release_date.split('-')[0]) if maoyan_release_date and len(maoyan_release_date) >= 4 else 0

    for idx, result_item in enumerate(search_results):
        link_element = result_item.xpath('.//h3/a')
        if not link_element:
            continue

        full_title_text = ''.join(link_element[0].xpath('.//text()')).strip()
        # Cleaned title for comparison: remove marks like [电影], (年份), etc.
        cleaned_title = re.sub(r'\[.*?\]\s*|\(\d{4}\)|\s*-?\s*.*?(?= \(豆瓣\))', '', full_title_text).strip()

        douban_result_year = 0
        subject_cast_elements = result_item.xpath('.//span[@class="subject-cast"]/text()')
        if subject_cast_elements:
            year_match = re.search(r'(\d{4})', subject_cast_elements[0])
            if year_match:
                douban_result_year = int(year_match.group(1))

        current_result_score = -1

        # Define matching criteria and assign scores
        if movie_name == cleaned_title and maoyan_year == douban_result_year:
            current_result_score = 3  # Best: Exact name and year
        elif movie_name == cleaned_title:
            current_result_score = 2  # Good: Exact name, year might differ or be missing
        elif (movie_name in cleaned_title or cleaned_title in movie_name) and maoyan_year == douban_result_year:
            current_result_score = 1  # Decent: Name containment and year matches
        elif movie_name in cleaned_title or cleaned_title in movie_name:
            current_result_score = 0  # Weak: Simple containment

        print(
            f"    DEBUG in get_data_douban: Result {idx + 1}: '{cleaned_title}' (Douban Year: {douban_result_year}), Current Score: {current_result_score}")

        if current_result_score > best_match_info['match_score']:
            best_match_info['match_score'] = current_result_score

            score_elements = result_item.xpath('.//div[@class="rating-info"]/span[@class="rating_nums"]/text()')
            best_match_info['score'] = score_elements[0].strip() if score_elements else '0.0'
            if best_match_info['score'] == '0.0':
                no_score_element = result_item.xpath('.//div[@class="rating-info"]/span[contains(text(), "暂无评分")]')
                if no_score_element:
                    best_match_info['score'] = '0.0'

            comment_text_elements = result_item.xpath(
                './/div[@class="rating-info"]/span[contains(text(), "人评价")]/text()')
            if comment_text_elements:
                comment_text = comment_text_elements[0].strip()
                comment_match = re.search(r'(\d+)', comment_text)
                best_match_info['comment'] = int(comment_match[0]) if comment_match else 0
            else:
                best_match_info['comment'] = 0

            href_attr = link_element[0].get('href')
            if href_attr:
                decoded_href_attr = unquote(href_attr)
                true_url_match = re.search(r'url=(https?://movie\.douban\.com/subject/\d+/)', decoded_href_attr)
                if true_url_match:
                    best_match_info['page_url'] = true_url_match.group(1)
                elif 'movie.douban.com/subject/' in decoded_href_attr:
                    best_match_info['page_url'] = decoded_href_attr
                else:
                    best_match_info['page_url'] = ''
            else:
                best_match_info['page_url'] = ''

            if current_result_score == 3:
                print(f"    DEBUG in get_data_douban: Found exact name and year match. Breaking search loop.")
                break

    print(
        f"  DEBUG in get_data_douban: Best match selected for '{movie_name}': score={best_match_info['score']}, comment={best_match_info['comment']}, page_url={best_match_info['page_url']}")
    return best_match_info['score'], best_match_info['comment'], best_match_info['page_url']


def sanitize_filename(name: str) -> str:
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, '_')
    return name.strip()


# --- 猫眼票房数据爬取与豆瓣详情信息整合 (get_data_maoyan) ---
def get_data_maoyan(maoyan_rankings_url: str) -> pd.DataFrame:
    all_data = []

    maoyan_html_file_path = "./html/maoyan_rankings_year.html"
    html_maoyan_rankings = _get_or_load_html(maoyan_rankings_url, maoyan_html_file_path, local_test)

    if not html_maoyan_rankings:
        print(f"Error: Failed to get HTML for Maoyan rankings from {maoyan_rankings_url}. Exiting get_data_maoyan.")
        return pd.DataFrame()

    movies_selector = lxml.html.fromstring(html_maoyan_rankings)
    rank_list_elements = movies_selector.xpath('//div[@id="ranks-list"]')
    if not rank_list_elements:
        print("Error: Could not find //div[@id='ranks-list'] on Maoyan page. Page structure might have changed.")
        return pd.DataFrame()
    rank_list = rank_list_elements[0]
    rows = rank_list.xpath('ul[@class="row"]')

    print(f"\n--- Processing Maoyan movies ({len(rows)} movies found) ---")

    for i, row in enumerate(rows):
        movie_name_maoyan = "未知电影"
        release_date = ""

        try:
            li_elements = row.xpath('li')
            if len(li_elements) < 5:
                print(f"  Warning: Maoyan row {i + 1} does not have enough 'li' elements. Skipping this row.")
                continue

            name_date_elements = li_elements[1].xpath('p')
            if len(name_date_elements) < 2:
                print(f"  Warning: Maoyan 'name_date' p elements are missing for row {i + 1}. Skipping.")
                continue

            movie_name_text = name_date_elements[0].xpath('text()')
            if movie_name_text:
                movie_name_maoyan = movie_name_text[0].strip()
            else:
                print(f"  Warning: Could not extract movie name from Maoyan row {i + 1}. Skipping.")
                continue

            date_text = name_date_elements[1].xpath('text()')
            release_date = date_text[0].strip().replace(' 上映', '') if date_text and date_text[0].strip() else ''

            total_box_office = 0
            money_text = li_elements[2].xpath('text()')
            try:
                if money_text and money_text[0].strip():
                    total_box_office = int(money_text[0].strip())
            except ValueError:
                print(f"  Warning: Could not parse total box office for '{movie_name_maoyan}'. Using 0.")

            avg_ticket_price = 0.0
            avg_money_text = li_elements[3].xpath('text()')
            try:
                if avg_money_text and avg_money_text[0].strip():
                    avg_ticket_price = float(avg_money_text[0].strip())
            except ValueError:
                print(f"  Warning: Could not parse average ticket price for '{movie_name_maoyan}'. Using 0.0.")

            avg_audience_per_screening = 0
            avg_people_text = li_elements[4].xpath('text()')
            try:
                if avg_people_text and avg_people_text[0].strip():
                    avg_audience_per_screening = int(avg_people_text[0].strip())
            except ValueError:
                print(f"  Warning: Could not parse average audience per screening for '{movie_name_maoyan}'. Using 0.")

            print(f"  Processing Maoyan movie {i + 1}: '{movie_name_maoyan}' (Box Office: {total_box_office}万元)")

            # --- 获取豆瓣搜索结果和详情页信息 ---
            # 拼接电影名和年份作为搜索关键词，使用“电影名 (年份)”格式
            search_query_with_year = movie_name_maoyan
            if release_date and len(release_date) >= 4:
                search_year = release_date.split('-')[0]
                search_query_with_year = f"{movie_name_maoyan} ({search_year})"  # 采用“电影名 (年份)”格式

            douban_search_url = get_search_url(search_query_with_year)  # 构造豆瓣搜索URL

            search_filename = sanitize_filename(search_query_with_year)[:50]
            douban_search_html_path = f"./html/douban_search_{search_filename}.html"

            html_search_result = _get_or_load_html(douban_search_url, douban_search_html_path, local_test)

            douban_score = '0.0'
            douban_comment_count = 0
            douban_detail_page_url = ''
            director_list, actor_list, movie_type_list, place_list, lang_list = [], [], [], [], []
            movie_year, movie_length = 0, 0

            if html_search_result:
                douban_score, douban_comment_count, douban_detail_page_url = get_data_douban(movie_name_maoyan,
                                                                                             html_search_result,
                                                                                             release_date)
                print(
                    f"    豆瓣搜索结果 (for '{movie_name_maoyan}'): 评分={douban_score}, 评论数={douban_comment_count}, 详情URL={douban_detail_page_url}")

                if douban_detail_page_url:
                    detail_filename_douban = sanitize_filename(movie_name_maoyan)[:50]
                    douban_detail_html_path = f"./html/douban_detail_{detail_filename_douban}_from_maoyan.html"

                    movie_html_detail = _get_or_load_html(douban_detail_page_url, douban_detail_html_path, local_test)

                    if movie_html_detail:
                        try:
                            director_list, actor_list, movie_type_list, place_list, lang_list, movie_year, movie_length = get_detail_data(
                                movie_html_detail)
                            print(
                                f"    Douban Detail Extracted for '{movie_name_maoyan}': Director={director_list}, Actor={actor_list}, Type={movie_type_list}, Place={place_list}, Lang={lang_list}, Year={movie_year}, Length={movie_length}")
                        except ValueError as ve:
                            print(f"    ValueError parsing Douban detail for '{movie_name_maoyan}': {ve}")
                        except Exception as e:
                            print(f"    Error parsing Douban detail for '{movie_name_maoyan}': {e}")
                    else:
                        print(f"    Warning: No HTML fetched for Douban detail page for '{movie_name_maoyan}'.")
                else:
                    print(f"    Warning: No valid Douban detail URL found for '{movie_name_maoyan}'.")
            else:
                print(f"  Warning: No HTML fetched for Douban search result for '{movie_name_maoyan}'.")

            # 组合所有数据 (注意这里移除了 movie_year，使其与db_store_2的期望参数数量匹配)
            data = [
                movie_name_maoyan, release_date, total_box_office, avg_ticket_price, avg_audience_per_screening,
                douban_score, douban_comment_count,
                director_list, actor_list, movie_type_list, place_list, lang_list, movie_length
            ]
            all_data.append(data)
            db_store_2(data)

            count += 1
            if count % 10 == 0:
                print(f"No.{count} Done: {movie_name_maoyan}")

        except Exception as e:
            print(f"  Error on processing Maoyan movie '{movie_name_maoyan}': {e}")
            continue

    if all_data:
        columns = ['电影名', '上映日期', '票房(万元)', '平均票价', '场均人数', '豆瓣评分', '豆瓣评论数',
                   '导演', '演员', '类型', '地区', '语言', '时长']
        all_data_df = pd.DataFrame(all_data, columns=columns)
        os.makedirs("./csv", exist_ok=True)
        all_data_df.to_csv('./csv/猫眼_豆瓣_整合数据.csv', index=False, encoding='utf-8-sig')
        print("猫眼_豆瓣整合数据已导出到 猫眼_豆瓣_整合数据.csv")
    else:
        print("No Maoyan and Douban combined data collected for CSV export.")

    return pd.DataFrame(all_data)


# --- 主程序入口 ---
if __name__ == "__main__":
    # --- 豆瓣TOP250数据爬取 (可选，默认不运行) ---
    # print("\n--- Starting Douban Top250 data collection ---")
    # get_data(douban_top250_url)
    # print("--- Douban Top250 data collection finished ---\n")

    # --- 猫眼票房及豆瓣搜索数据爬取 ---
    print("\n--- Starting Maoyan and Douban search data collection ---")
    result_maoyan = get_data_maoyan(maoyan_rankings_url)
    print("--- Maoyan and Douban search data collection finished ---")

    # 确保数据库连接在程序结束时关闭
    try:
        from database import db

        if db:
            db.close()
            print("\nDatabase connection closed.")
    except Exception as e:
        print(f"Error closing database connection: {e}")