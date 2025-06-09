# movie_parser.py

import lxml.html
import re
import json
from urllib.parse import unquote


def parse_maoyan_rankings(html_content: str) -> list:
    """【猫眼票房榜】解析"""
    if not html_content: return []
    selector = lxml.html.fromstring(html_content)
    movies_data, rows = [], selector.xpath('//div[@id="ranks-list"]/ul[@class="row"]')
    if not rows: return []
    for row in rows:
        try:
            name = row.xpath('./li[2]/p[1]/text()')[0].strip()
            release_date = row.xpath('./li[2]/p[2]/text()')[0].strip().replace(' 上映', '')
            box_office_str = row.xpath('./li[3]/text()')[0].strip()
            box_office = float(re.search(r'(\d+\.?\d*)', box_office_str).group(1)) * (
                10000 if '亿' in box_office_str else 1)
            avg_price = float(row.xpath('./li[4]/text()')[0].strip())
            avg_audience = int(row.xpath('./li[5]/text()')[0].strip())
            movies_data.append(
                {"name": name, "release_date": release_date, "box_office": box_office, "avg_price": avg_price,
                 "avg_audience": avg_audience})
        except Exception:
            continue
    return movies_data


def find_best_match_from_douban_search(html_content: str) -> dict:
    """【豆瓣搜索页】直接获取第一个电影结果"""
    if not html_content: return {}
    selector = lxml.html.fromstring(html_content)
    first_result = selector.xpath(
        '(//div[@class="result-list"]/div[contains(@class, "result") and .//span[contains(text(), "电影")]])[1]')
    if not first_result: return {}
    item = first_result[0]
    url_elements = item.xpath('.//h3/a/@href')
    if not url_elements: return {}
    true_url_match = re.search(r'url=(https?://movie\.douban\.com/subject/\d+/)', unquote(url_elements[0]))
    if not true_url_match: return {}
    douban_url = true_url_match.group(1)
    douban_id = int(re.search(r'/subject/(\d+)/', douban_url).group(1))
    score_elements = item.xpath('.//span[@class="rating_nums"]/text()')
    douban_score = float(score_elements[0]) if score_elements else 0.0
    comment_elements = item.xpath('.//span[contains(text(), "人评价")]/text()')
    comment_match = re.search(r'(\d+)', comment_elements[0]) if comment_elements else None
    douban_comment_count = int(comment_match.group(1)) if comment_match else 0
    return {"douban_id": douban_id, "douban_url": douban_url, "douban_score": douban_score,
            "douban_comment_count": douban_comment_count}


def get_douban_detail_data(html_content: str) -> dict:
    """
    【最终修复版】精准解析豆瓣详情页所有信息。
    """
    if not html_content: return {}

    selector = lxml.html.fromstring(html_content)

    details = {
        "name_en": "", "quote": "-", "directors": [], "actors": [], "genres": [], "regions": [], "languages": [],
        "release_date_text": "", "runtime_text": "", "year": 0, "cover_url": "",
        "synopsis": "", "comments_json": "[]"
    }

    # --- 【核心修复】外文名 (name_en) ---
    # 外文名和中文名在同一个<span>里
    title_text_elements = selector.xpath('//h1/span[@property="v:itemreviewed"]/text()')
    if title_text_elements:
        full_title = title_text_elements[0].strip()
        # 假设第一个空格之后的就是外文名
        parts = full_title.split(' ', 1)
        if len(parts) > 1:
            details["name_en"] = parts[1].strip()

    # --- 核心信息块 <div id="info"> ---
    info_div_list = selector.xpath('//div[@id="info"]')
    if not info_div_list: return details
    info_div = info_div_list[0]
    info_text = info_div.xpath('string(.)')

    details["directors"] = [d.strip() for d in info_div.xpath('.//a[@rel="v:directedBy"]/text()')]
    details["actors"] = [a.strip() for a in info_div.xpath('.//a[@rel="v:starring"]/text()')][:10]
    details["genres"] = [g.strip() for g in info_div.xpath('.//span[@property="v:genre"]/text()')]
    details["release_date_text"] = ' / '.join(info_div.xpath('.//span[@property="v:initialReleaseDate"]/text()'))
    runtime_elements = info_div.xpath('.//span[@property="v:runtime"]/text()')
    details["runtime_text"] = runtime_elements[0].strip() if runtime_elements else ''

    region_match = re.search(r'制片国家/地区: (.*?)\n', info_text)
    if region_match: details["regions"] = [r.strip() for r in region_match.group(1).split('/')]
    lang_match = re.search(r'语言: (.*?)\n', info_text)
    if lang_match: details["languages"] = [l.strip() for l in lang_match.group(1).split('/')]
    year_match = re.search(r'(\d{4})', details["release_date_text"])
    if year_match: details["year"] = int(year_match.group(1))

    # --- 页面其他部分信息 ---
    cover_url_elements = selector.xpath('//div[@id="mainpic"]//img/@src')
    if cover_url_elements: details["cover_url"] = cover_url_elements[0]

    synopsis_span = selector.xpath('//div[@id="link-report-intra"]//span[@property="v:summary"]')
    if synopsis_span:
        details["synopsis"] = synopsis_span[0].text_content().strip()
    else:
        all_hidden_span = selector.xpath('//div[@id="link-report-intra"]//span[@class="all hidden"]')
        if all_hidden_span: details["synopsis"] = all_hidden_span[0].text_content().strip()

    hot_comments_list = []
    hot_comments_section = selector.xpath('//div[@id="hot-comments"]')
    if hot_comments_section:
        comment_items = hot_comments_section[0].xpath('.//div[@class="comment-item"]')
        for item in comment_items:
            try:
                author = item.xpath('.//span[@class="comment-info"]/a/text()')[0].strip()
                comment = item.xpath('.//p[contains(@class, "comment-content")]/span[@class="short"]/text()')[0].strip()
                if author and comment:
                    hot_comments_list.append({"author": author, "comment": comment})
            except IndexError:
                continue
    details["comments_json"] = json.dumps(hot_comments_list, ensure_ascii=False)

    # 语录(quote)在详情页不存在，所以details['quote']会保持默认值'-'

    return details