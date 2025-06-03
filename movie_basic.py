# movie_basic.py

import lxml.html
import re
from lxml.html import tostring # 确保这一行还在，因为你后续使用了tostring

def get_basic_data(movie):
    # 电影信息
    movie_info = movie.xpath('div[@class="info"]')
    if not movie_info: return ('-',)*6 # 如果找不到 info 节点，直接返回默认值
    movie_info = movie_info[0]

    movie_bd = movie_info.xpath('div[@class="bd"]')
    if not movie_bd: return ('-',)*6
    movie_bd = movie_bd[0]

    # --- 获取电影名 (保持原逻辑，使用tostring和正则) ---
    movie_div = tostring(movie_info, encoding="utf-8").decode("utf-8")

    movie_title = re.findall(r'<span class="title">(.*)</span>', movie_div)
    name1 = movie_title[0].strip() if movie_title else '-' # 检查列表是否为空
    if len(movie_title) > 1:
        name2 = "".join(movie_title[1].strip()[1:].split())
    else:
        name2 = '-'

    # --- 电影评分 (根据截图修正 XPath) ---
    # 评分现在位于 div[@class="bd"] 下面的无 class div 内部的 span[@class="rating_num"]
    score_elements = movie_bd.xpath('div/span[@class="rating_num"]')
    if score_elements:
        score = float(score_elements[0].text)
    else:
        score = 0.0 # 默认评分

    # --- 评价人数 (根据截图修正 XPath 和提取逻辑) ---
    # 评价人数位于 div[@class="bd"] 下面的无 class div 内部的最后一个 span (包含“人评价”文本)
    comment_text_elements = movie_bd.xpath('div/span[contains(text(), "人评价")]')
    if comment_text_elements:
        comment_text = comment_text_elements[0].text
        comment_match = re.findall(r'\d+', comment_text)
        comment = int(comment_match[0]) if comment_match else 0
    else:
        comment = 0 # 默认评价人数

    # --- 电影语录 (保持原逻辑，增加健壮性) ---
    movie_quote = movie_bd.xpath('p[@class="quote"]')
    if len(movie_quote) > 0:
        quote_span = movie_quote[0].xpath('span/text()')
        quote_str = quote_span[0] if quote_span else '' # 确保span有文本
    else:
        quote_str = ''

    # --- 电影详情URL (根据截图修正正则匹配) ---
    # 这里的URL应该是在 div class="info" 的 div class="hd" 下面的 a 标签的 href 属性
    # 原正则表达式的 class="" 是错的。直接用 XPath 更稳妥。
    page_url_elements = movie_info.xpath('div[@class="hd"]/a/@href')
    if page_url_elements:
        page_url = page_url_elements[0]
    else:
        page_url = '' # 默认URL


    return name1, name2, score, comment, quote_str, page_url