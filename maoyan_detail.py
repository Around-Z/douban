import lxml.html
import re
import json


def get_maoyan_detail_data(maoyan_html: str):
    """
    解析猫眼电影详情页的HTML内容，提取电影的详细信息。

    参数:
        maoyan_html (str): 猫眼电影详情页的HTML内容。

    返回:
        tuple: (director, actor, movie_type, place, lang, length)
    """
    _selector = lxml.html.fromstring(maoyan_html)

    # 初始化所有返回值，确保即使提取失败也返回正确类型
    director = []
    actor = []
    movie_type = []
    place = []
    lang = []  # 猫眼详情页通常没有明确列出语言
    length = 0

    # 1. 尝试从 <script id="pageData"> JSON 中提取数据（优先选择，结构化）
    page_data_script = _selector.xpath('//script[@id="pageData"]/text()')
    if page_data_script:
        try:
            page_data = json.loads(page_data_script[0].strip())

            # 导演
            if 'director' in page_data and page_data['director']:
                director = [d.strip() for d in page_data['director'].split(',') if d.strip()]

            # 类型
            if 'category' in page_data and page_data['category']:
                movie_type = [t.strip() for t in page_data['category'].split(',') if t.strip()]

        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse pageData JSON in maoyan_detail.py: {e}")

    # 2. 从 HTML 结构中提取数据（用于 JSON 中没有的字段，或作为备用）

    # 地区 和 时长 - 通常在同一个P标签中，例如 "中国大陆 / 176分钟 "
    info_source_duration_elements = _selector.xpath(
        '//div[@class="info-source-duration"]/div/p[@class=".ellipsis-1"]/text()')
    if info_source_duration_elements:
        combined_info = info_source_duration_elements[0].strip()
        parts = combined_info.split('/')

        # 地区
        if len(parts) > 0 and parts[0].strip():
            place = [parts[0].strip()]

        # 时长
        if len(parts) > 1 and parts[1].strip():
            length_str = parts[1].strip()
            length_match = re.search(r'(\d+)', length_str)
            if length_match:
                length = int(length_match.group(1))

    # 演员 (猫眼详情页的演员列表通常在HTML中，不在pageData JSON中)
    # 结构: <div class="movie-person-item"><p class="name">姓名</p><p class="role">角色</p></div>
    # 我们查找 role 包含 "饰" 或 "演员" 的项
    actor_elements = _selector.xpath(
        '//div[@class="movie-person-list"]//div[p[@class="role" and (contains(text(), "饰") or contains(text(), "演员"))]]/a[@class="actor-name J_actorName"]/text()')
    actor = [a.strip() for a in actor_elements if a.strip()]

    # 如果导演没有从JSON中获取到，则从HTML列表获取
    if not director:
        director_elements = _selector.xpath(
            '//div[@class="movie-person-list"]//div[p[@class="role" and contains(text(), "导演")]]/a[@class="actor-name J_actorName"]/text()')
        director = [d.strip() for d in director_elements if d.strip()]

    return director, actor, movie_type, place, lang, length