# modules/page_artistry.py (最终交互与资源修复版)

import streamlit as st
import pandas as pd
import json
import os  # 导入os模块以检查文件路径
from .visualizations import generate_wordcloud_text, create_and_show_wordcloud, get_proxied_image_url


# --- 详情页渲染函数 ---
# modules/page_artistry.py

# ... (imports 保持不变) ...

# --- 【最终修复与功能补全版】详情页渲染函数 ---
def render_single_movie_details(df_top250):
    """
    渲染单片深度解读模块。
    此版本恢复了“影迷心声”和“艺术关联推荐”模块。
    """
    # 1. 从session_state获取ID
    douban_id = st.session_state.get('selected_movie_douban_id')
    if not douban_id:
        return

    # 2. 返回按钮逻辑
    if st.button("⬅️ 返回海报墙"):
        st.session_state.selected_movie_douban_id = None
        st.rerun()

    # 3. 安全地获取电影行
    movie_series = df_top250[df_top250['douban_id'] == douban_id]
    if movie_series.empty:
        st.error(f"错误：在数据中找不到 douban_id 为 {douban_id} 的电影。")
        return
    movie = movie_series.iloc[0]

    # --- 4. 详情页UI渲染 (上半部分) ---
    st.title(f"{movie.get('name_cn', '未知电影')} ({movie.get('name_en', '')})")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(get_proxied_image_url(movie.get('cover_url')), use_container_width=True)

    with col2:
        c1, c2 = st.columns(2)
        c1.metric("豆瓣评分", f"{movie.get('score', 0.0):.1f} ⭐")
        c2.metric("评价人数", f"{int(movie.get('comment_count', 0)):,}")

        st.markdown(f"**导演**: {', '.join(movie.get('directors_list', [])) or '暂无数据'}")
        st.markdown(f"**类型**: {', '.join(movie.get('genres_list', [])) or '未知'}")
        st.markdown(
            f"**上映年份**: {int(movie.get('year')) if pd.notna(movie.get('year')) and movie.get('year') > 0 else '未知'}")

        if pd.notna(movie.get('quote')):
            st.markdown(f"> {movie.get('quote')}")

        with st.expander("查看剧情简介"):
            st.write(movie.get('synopsis', '暂无简介。'))

    st.markdown("---")  # 添加分割线

    # --- 5. 【恢复】影迷心声 (热门评论) 模块 ---
    st.subheader("💬 影迷心声")
    # 此处假设 hot_comments_json 列在 data_loader.py 中已被正确加载
    try:
        # hot_comments_json 是一个JSON字符串，需要用json.loads解析
        comments_str = movie.get('hot_comments_json')
        if pd.notna(comments_str):
            comments_list = json.loads(comments_str)
            if comments_list:
                for comment in comments_list[:5]:  # 只显示前5条
                    st.info(f"**{comment.get('author', '豆友')}**: {comment.get('comment', '')}")
            else:
                st.info("暂无热门评论。")
        else:
            st.info("暂无热门评论。")
    except Exception as e:
        st.warning(f"热门评论解析失败，可能格式有误。错误: {e}")
        st.info("暂无热门评论。")

    st.markdown("---")  # 添加分割线

    # --- 6. 【恢复】艺术关联推荐 模块 ---
    st.subheader("🧭 探索更多")

    directors = movie.get('directors_list', [])
    if directors:
        related_movies = df_top250[
            (df_top250['directors_list'].apply(lambda x: any(d in x for d in directors))) &
            (df_top250['douban_id'] != douban_id)
            ]
        if not related_movies.empty:
            st.markdown("**师出同门:**")
            # 使用更小的列数以避免图片过小
            cols = st.columns(min(5, len(related_movies)))
            for i, (_, rel_movie) in enumerate(related_movies.head(5).iterrows()):
                with cols[i]:
                    st.image(get_proxied_image_url(rel_movie['cover_url']), caption=rel_movie['name_cn'])
        else:
            st.info("在本库中未找到该导演的其他作品。")
    else:
        st.info("由于缺少导演数据，无法进行“师出同门”推荐。")


# ... render_page_artistry 函数保持不变 ...

# --- 主页面渲染函数 ---
def render_page_artistry(df_top250):
    if st.session_state.get('selected_movie_douban_id') is not None:
        render_single_movie_details(df_top250);
        return

    st.title("🏛️ 艺术殿堂")
    st.caption("豆瓣电影Top250全景画卷")
    st.markdown("---")

    st.header("殿堂主题：剧情基因图谱")
    col1, col_center, col3 = st.columns([0.1, 3, 0.1])
    with col_center:
        with st.spinner("正在绘制剧情基因图谱..."):
            processed_text = generate_wordcloud_text(df_top250['synopsis'], "assets/stopwords.txt")

            # --- 【最终调用方式】只传递两个参数 ---
            create_and_show_wordcloud(
                processed_text,
                colormap='cividis'  # 你可以在这里切换色盘，如 'gist_heat'
            )

    st.markdown("---")
    st.header("经典海报墙")
    sorted_df = df_top250.sort_values('ranking')
    movies_per_row = 6

    for i in range(0, len(sorted_df), movies_per_row):
        cols = st.columns(movies_per_row)
        for j, col in enumerate(cols):
            if i + j < len(sorted_df):
                movie = sorted_df.iloc[i + j]
                with col:
                    st.image(get_proxied_image_url(movie.get('cover_url')), use_container_width=True)

                    # 使用 douban_id 生成唯一的key
                    if st.button(f"#{int(movie.get('ranking', 0))} {movie.get('name_cn', '')}",
                                 key=f"btn_movie_{movie['douban_id']}",
                                 use_container_width=True):
                        # 点击按钮时，设置session_state并触发重绘
                        st.session_state['selected_movie_douban_id'] = movie['douban_id']
                        st.rerun()