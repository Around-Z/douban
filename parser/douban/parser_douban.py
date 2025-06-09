# parser_douban.py

import lxml.html
import re
import json


def get_basic_data(item_selector):
    """【豆瓣Top250列表页】解析单个电影条目"""
    try:
        name_elements = item_selector.xpath('.//span[@class="title"][1]/text()')
        name1 = name_elements[0].strip() if name_elements else '未知电影'
        url_elements = item_selector.xpath('.//div[@class="hd"]/a/@href')
        page_url = url_elements[0] if url_elements else ''
        douban_id_match = re.search(r'/subject/(\d+)/', page_url)
        douban_id = int(douban_id_match.group(1)) if douban_id_match else 0
        name2_elements = item_selector.xpath('.//span[@class="title"][2]/text()')
        name2 = name2_elements[0].replace('\xa0/\xa0', '').strip() if name2_elements else ''
        score_elements = item_selector.xpath('.//span[@class="rating_num"]/text()')
        score = score_elements[0] if score_elements else '0.0'
        # comment_elements = item_selector.xpath('.//div[@class="star"]/span[4]/text()')
        comment_elements = item_selector.xpath('.//span[contains(text(), "人评价")]/text()')
        comment_text = comment_elements[0] if comment_elements else ''
        comment_match = re.search(r'(\d+)', comment_text)
        # comment_elements = item.xpath('.//span[contains(text(), "人评价")]/text()')
        # comment_match = re.search(r'(\d+)', comment_elements[0]) if comment_elements else None
        # douban_comment_count = int(comment_match.group(1)) if comment_match else 0
        comment_count = comment_match.group(1) if comment_match else '0'
        quote_elements = item_selector.xpath('.//p[@class="quote"]/span/text()')
        quote_str = quote_elements[0].strip() if quote_elements else '-'
        # quote_elements = item_selector.xpath('.//p[@class="quote"]/span[@class="inq"]/text()')
        # quote_str = quote_elements[0].strip() if quote_elements else '-'
    except Exception as e:
        print(f"  - 解析Top250列表页条目时出错: {e}")
        return '错误电影', '', '0.0', '0', '-', '', 0
    return name1, name2, score, comment_count, quote_str, page_url, douban_id


def get_douban_detail_data(html_content: str) -> dict:
    """【豆瓣详情页】解析所有详细信息"""
    if not html_content: return {}
    selector = lxml.html.fromstring(html_content)

    details = {
        "name_en": "", "directors": [], "actors": [], "genres": [], "regions": [],
        "languages": [], "release_date_text": "", "runtime_text": "", "year": 0,
        "cover_url": "", "synopsis": "", "comments_json": "[]"
    }

    title_elements = selector.xpath('//h1/span[@property="v:itemreviewed"]/text()')
    if title_elements:
        parts = title_elements[0].strip().split(' ', 1)
        if len(parts) > 1: details["name_en"] = parts[1].strip()

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

    cover_url_elements = selector.xpath('//div[@id="mainpic"]//img/@src')
    if cover_url_elements: details["cover_url"] = cover_url_elements[0]

    synopsis_span = selector.xpath('//div[@id="link-report-intra"]//span[@property="v:summary"]')
    if synopsis_span: details["synopsis"] = synopsis_span[0].text_content().strip()

    hot_comments_list = []
    hot_comments_section = selector.xpath('//div[@id="hot-comments"]')
    if hot_comments_section:
        comment_items = hot_comments_section[0].xpath('.//div[@class="comment-item"]')
        for item in comment_items:
            try:
                author = item.xpath('.//span[@class="comment-info"]/a/text()')[0].strip()
                comment = item.xpath('.//p[contains(@class, "comment-content")]/span[@class="short"]/text()')[0].strip()
                if author and comment: hot_comments_list.append({"author": author, "comment": comment})
            except IndexError:
                continue
    details["comments_json"] = json.dumps(hot_comments_list, ensure_ascii=False)

    return details