# modules/page_directors.py (V3.1 - 视觉与交互终极版)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
# 确保从visualizations导入了get_proxied_image_url
# 如果你的visualizations.py还没有这个函数，请添加
from .visualizations import get_proxied_image_url


# --- 核心数据与标签准备函数 ---
@st.cache_data
def prepare_director_cosmos_data(df_full):
    """
    为“导演星空图”准备数据，为每位导演打上“星空”标签。
    """
    # 确保必要列存在
    required_cols = ['directors_list', 'name_cn', 'score', 'is_top250']
    if not all(col in df_full.columns for col in required_cols):
        st.error("缺少必要的列来进行导演分析，请检查数据加载过程。")
        return pd.DataFrame(), pd.DataFrame()

    # 安全地处理可选列
    if 'runtime_text' in df_full.columns:
        df_full['runtime_minutes'] = pd.to_numeric(df_full['runtime_text'].str.extract(r'(\d+)')[0],
                                                   errors='coerce').fillna(0)
    else:
        df_full['runtime_minutes'] = 0

    if 'box_office_ten_thousand' not in df_full.columns:
        df_full['box_office_ten_thousand'] = 0

    df_director_movie = df_full.explode('directors_list').rename(columns={'directors_list': 'director_name'})
    df_director_movie = df_director_movie.dropna(subset=['director_name'])

    grouped_by_director = df_director_movie.groupby('director_name')
    df_stats = grouped_by_director.agg(
        电影数量=('name_cn', 'count'),
        总票房_万元=('box_office_ten_thousand', 'sum'),
        平均票房_万元=('box_office_ten_thousand', 'mean'),
        平均评分=('score', 'mean'),
        Top250作品数=('is_top250', 'sum')
    ).reset_index()

    # 标签定义
    box_office_threshold = df_full['box_office_ten_thousand'].nlargest(50).min() if not df_full[
        'box_office_ten_thousand'].empty and df_full['box_office_ten_thousand'].nlargest(50).shape[0] > 0 else 0
    is_box_giant = df_director_movie[
                       'box_office_ten_thousand'] >= box_office_threshold if box_office_threshold > 0 else False
    box_giants = set(df_director_movie[is_box_giant]['director_name'])

    is_prestige = df_director_movie['is_top250'] == True
    prestige_stars = set(df_director_movie[is_prestige]['director_name'])

    def get_star_type(row):
        is_p = row['director_name'] in prestige_stars
        is_b = row['director_name'] in box_giants
        if is_p and is_b: return "双子星"
        if is_p: return "口碑之星"
        if is_b: return "票房巨星"
        return "星尘"

    df_stats['star_type'] = df_stats.apply(get_star_type, axis=1)

    return df_stats.set_index('director_name'), df_director_movie


# --- 王者殿堂渲染函数 ---
def render_hall_of_kings(king_type, df_stats, df_director_movie):
    background_images = {
        "box_office": "https://images.unsplash.com/photo-1593814681462-e61884b6e428?q=80&w=2070",
        "prestige": "https://images.unsplash.com/photo-1531945399622-30509a282381?q=80&w=2070"
    }
    bg_image_url = background_images.get(king_type)
    st.markdown(
        f"""<style>.stApp {{ background-image: url("{bg_image_url}"); background-size: cover; background-attachment: fixed; }}</style>""",
        unsafe_allow_html=True)

    if st.button("⬅️ 返回星空图", key="back_from_hall"):
        st.session_state.king_mode = None
        st.rerun()

    if king_type == "box_office":
        king_director = df_stats.nlargest(1, '总票房_万元').index[0]
        king_data = df_stats.loc[king_director]
        director_movies = df_director_movie[df_director_movie['director_name'] == king_director]
        st.markdown(
            f"<h1 style='text-align:center; color:gold; text-shadow: 2px 2px 8px #000;'>票房之王：{king_director}</h1>",
            unsafe_allow_html=True)
        total_box_office_yi = king_data['总票房_万元'] / 10000
        counter_placeholder = st.empty()
        for i in np.linspace(0, total_box_office_yi, num=50):
            counter_placeholder.markdown(
                f"<h2 style='text-align:center; color:white;'>累计总票房: <span style='color:gold; font-size:2em;'>{i:.2f} 亿</span></h2>",
                unsafe_allow_html=True)
            time.sleep(0.02)
        counter_placeholder.markdown(
            f"<h2 style='text-align:center; color:white;'>累计总票房: <span style='color:gold; font-size:2em;'>{total_box_office_yi:.2f} 亿</span></h2>",
            unsafe_allow_html=True)
        st.subheader("票房瀑布图")
        waterfall_df = director_movies.sort_values('release_date')
        fig = go.Figure(go.Waterfall(name="票房累积", orientation="v", measure=["relative"] * len(waterfall_df),
                                     x=waterfall_df['name_cn'].tolist(),
                                     text=[f"{b / 10000:.2f}亿" for b in waterfall_df['box_office_ten_thousand']],
                                     y=waterfall_df['box_office_ten_thousand'].tolist()))
        fig.update_layout(title="生涯票房累积之路", showlegend=False, yaxis_title="票房 (万元)")
        st.plotly_chart(fig, use_container_width=True)

    elif king_type == "prestige":
        king_director = df_stats[df_stats.get('电影数量', 0) >= 3].nlargest(1, '平均评分').index[0]
        king_data = df_stats.loc[king_director]
        st.markdown(
            f"<h1 style='text-align:center; color:#E0E0E0; text-shadow: 1px 1px 5px #000;'>口碑之王：{king_director}</h1>",
            unsafe_allow_html=True)
        avg_score = king_data['平均评分']
        stars = "⭐" * int(avg_score / 2) + ("✨" if avg_score % 2 >= 0.5 else "")
        st.markdown(
            f"<h2 style='text-align:center; color:white;'>平均口碑: <span style='color:#17A2B8; font-size:2em;'>{stars}</span> {avg_score:.2f}/10</h2>",
            unsafe_allow_html=True)

        st.subheader("荣誉长廊 (Top250作品)")
        honor_movies = df_director_movie[(df_director_movie['director_name'] == king_director) & (
                    df_director_movie['is_top250'] == True)].sort_values('ranking')
        if honor_movies.empty:
            st.info(f"导演 {king_director} 的Top250作品未在当前数据集中找到。")
            return

        cards_html = ""
        for _, movie in honor_movies.iterrows():
            img_src = get_proxied_image_url(movie.get('cover_url'))
            ranking, name, score, quote = int(movie.get('ranking', 0)), movie.get('name_cn',
                                                                                  '未知'), f"{movie.get('score', 0):.1f}", movie.get(
                'quote')

            badge_html = f"<div style='background-color:#FFD700;color:black;border-radius:50%;width:50px;height:50px;display:flex;align-items:center;justify-content:center;font-weight:bold;font-size:1.2em;flex-shrink:0;'>#{ranking}</div>"
            score_html = f"<div style='margin-left:15px;'><div style='font-size:0.8em;color:#aaa;'>豆瓣评分</div><div style='font-size:1.8em;font-weight:bold;color:white;'>{score}</div></div>"
            quote_html = f"<div style='background-color:rgba(0,150,255,0.1);border-left:3px solid #00BFFF;padding:10px;margin-top:15px;font-style:italic;font-size:0.9em;'>“{quote}”</div>" if pd.notna(
                quote) and quote.strip() else ""
            card_html = f"""<div style="flex:0 0 280px;background-color:rgba(40,40,40,0.8);border-radius:10px;padding:15px;border:1px solid #444;margin-right:15px;color:white;"><img src="{img_src}" style="width:100%;border-radius:5px;"><h4 style="margin-top:15px;margin-bottom:15px;height:3.2em;overflow:hidden;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;">{name}</h4><div style="display:flex;align-items:center;">{badge_html}{score_html}</div>{quote_html}</div>"""
            cards_html += card_html
        st.markdown(f"<div style='display:flex;overflow-x:auto;padding:10px 0;'>{cards_html}</div>",
                    unsafe_allow_html=True)


# --- 页面主渲染函数 ---
def render_page_directors(df_full):
    if 'king_mode' not in st.session_state: st.session_state.king_mode = None
    df_stats, df_director_movie = prepare_director_cosmos_data(df_full)
    if df_stats.empty: st.warning("没有足够的数据来构建导演星空。"); return

    if st.session_state.king_mode:
        render_hall_of_kings(st.session_state.king_mode, df_stats, df_director_movie)
        return

    st.title("🎬 导演星空图 (Director's Cosmos)")
    st.info("每一位导演都是夜空中的一颗星。在此探索由他们构成的璀璨星座。")

    col1, col2 = st.columns([3, 1])
    with col1:
        plot_df = df_stats.copy().reset_index()
        plot_df['log_平均票房'] = np.log10(plot_df['平均票房_万元'] + 1)
        plot_df['总票房_亿'] = plot_df['总票房_万元'] / 10000

        custom_data = plot_df[['director_name', 'star_type', '电影数量', '平均评分', '总票房_亿']]
        hover_template = (
            "<b>%{customdata[0]}</b><br>"
            "<span style='font-size: 12px; color: #999;'><i>%{customdata[1]}</i></span>"
            # 使用<br>和Unicode下划线字符来创建分割线
            "<br>――――――――――――――――――<br>"
            "作品数: <b>%{customdata[2]}</b><br>"
            "平均评分: <b style='color: #17A2B8;'>%{customdata[3]:.1f}</b><br>"
            "总票房: <b style='color: #FFC107;'>%{customdata[4]:.1f} 亿</b>"
            "<extra></extra>"  # 隐藏多余的trace信息
        )

        category_orders, color_map = {"star_type": ["双子星", "票房巨星", "口碑之星", "星尘"]}, {"双子星": "#9467bd",
                                                                                                 "票房巨星": "#ffd700",
                                                                                                 "口碑之星": "#1f77b4",
                                                                                                 "星尘": "#7f7f7f"}

        fig_cosmos = px.scatter(
            plot_df, x="平均评分", y="log_平均票房", size="总票房_万元", color="star_type", custom_data=custom_data,
            category_orders=category_orders, color_discrete_map=color_map, size_max=60,
            labels={"平均评分": "口碑维度", "log_平均票房": "商业维度(对数)", "star_type": "星座"}
        )
        fig_cosmos.update_traces(hovertemplate=hover_template)
        fig_cosmos.update_layout(
            height=600, legend_title_text='点击图例筛选', font_family="SimHei, Microsoft YaHei, sans-serif",
            hoverlabel=dict(bgcolor="rgba(14,17,23,0.9)", font_size=14, bordercolor="rgba(255,215,0,0.5)")
        )
        st.plotly_chart(fig_cosmos, use_container_width=True)

    with col2:
        st.subheader("👑 王者殿堂")
        if st.button("查看票房之王", use_container_width=True):
            st.session_state.king_mode = "box_office";
            st.rerun()
        if st.button("查看口碑之王", use_container_width=True):
            st.session_state.king_mode = "prestige";
            st.rerun()

    st.header("👤 单星闪耀：导演聚焦")
    director_list = sorted(df_stats.index.tolist())
    selected_director = st.selectbox("选择一颗星辰进行聚焦:", director_list)
    if selected_director:
        st.dataframe(df_stats.loc[selected_director])
        st.write(f"**{selected_director} 的作品列表:**")
        st.dataframe(df_director_movie[df_director_movie['director_name'] == selected_director][
                         ['name_cn', 'year', 'score', 'box_office_ten_thousand']])