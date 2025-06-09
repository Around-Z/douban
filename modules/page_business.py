# modules/page_business.py (专业增强版 V3.2)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
# 导入创建分布图/脊线图的工厂函数
from plotly.figure_factory import create_distplot
# 导入我们自己的可视化辅助函数
from .visualizations import generate_wordcloud_text, create_and_show_wordcloud


# --- 页面主渲染函数 ---
def render_page_business(df_full):
    """渲染“商业版图”页面的主函数，包含所有专业级可视化升级。"""
    st.title("📈 商业版图：电影市场深度剖析")

    # --- 全局筛选器 ---
    analysis_scope = st.radio(
        "分析对象:",
        ("所有电影", "仅分析豆瓣Top250"),
        horizontal=True,
        key="business_scope"
    )

    if analysis_scope == "仅分析豆瓣Top250":
        df_analysis = df_full[df_full['is_top250'] == True].copy()
    else:
        df_analysis = df_full.copy()

    if df_analysis.empty:
        st.warning(f"在 '{analysis_scope}' 范围内没有可供分析的数据。")
        return

    # --- 创建选项卡 ---
    tab1, tab2, tab3 = st.tabs(["📊 市场宏观趋势", "🧬 内容基因解码", "❤️ 口碑与票房关系"])

    # --- 选项卡1: 市场宏观趋势 (视觉升级) ---
    with tab1:
        st.header("历年市场构成演进图")
        st.info("观察不同类型电影对总票房贡献的结构性变化。")
        df_plot = df_analysis.dropna(subset=['year', 'box_office_ten_thousand', 'genres_list'])
        if not df_plot.empty:
            df_plot['main_genre'] = df_plot['genres_list'].apply(lambda x: x[0] if x else '其他')
            top_genres = df_plot['main_genre'].value_counts().nlargest(5).index
            df_plot['main_genre_agg'] = df_plot['main_genre'].apply(lambda x: x if x in top_genres else '其他')

            df_area = df_plot.groupby(['year', 'main_genre_agg'])['box_office_ten_thousand'].sum().reset_index()
            fig_area = px.area(df_area, x='year', y='box_office_ten_thousand', color='main_genre_agg',
                               title="历年各类型电影票房贡献堆叠面积图",
                               labels={'box_office_ten_thousand': '票房(万元)'})
            st.plotly_chart(fig_area, use_container_width=True)

        st.header("档期票房热力图")
        st.info("探索“黄金档期”：颜色越深，代表该“月份-星期”组合的平均票房越高。")
        df_heatmap = df_analysis.dropna(subset=['month', 'dayofweek', 'box_office_ten_thousand'])
        if not df_heatmap.empty:
            # 创建数据透视表
            pivot_table = df_heatmap.pivot_table(
                values='box_office_ten_thousand',
                index='month',
                columns='dayofweek',
                aggfunc='mean'
            )

            # 【修复】使用 reindex 保证热力图有完整的星期列 (0-6)
            all_weekdays = range(7)
            pivot_table = pivot_table.reindex(columns=all_weekdays).fillna(0)
            pivot_table.columns = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']

            fig_heatmap = px.imshow(pivot_table, text_auto=".0f", aspect="auto", color_continuous_scale='Viridis',
                                    title="月份-星期几 平均票房(万元)热力图")
            st.plotly_chart(fig_heatmap, use_container_width=True)

    # --- 选项卡2: 内容基因解码 (视觉升级) ---
    with tab2:
        st.header("叙事DNA对比：票房巨头 vs 口碑神作")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("票房巨头 (Top 100)")
            text_boxoffice = generate_wordcloud_text(df_full.nlargest(100, 'box_office_ten_thousand')['synopsis'])
            create_and_show_wordcloud(text_boxoffice, background_color="white")
        with col2:
            st.subheader("口碑神作 (Top 100)")
            text_score = generate_wordcloud_text(df_full.nlargest(100, 'score')['synopsis'])
            create_and_show_wordcloud(text_score, background_color="white")

        st.header("类型投资回报象限图 (Genre ROI Quadrant)")
        st.info("探索电影类型的投资价值：X轴代表竞争激烈程度，Y轴代表平均盈利能力。")
        df_genres_exploded = df_analysis.explode('genres_list').dropna(subset=['genres_list'])
        df_genre_stats = df_genres_exploded.groupby('genres_list').agg(
            电影数量=('name_cn', 'count'),
            平均票房=('box_office_ten_thousand', 'mean'),
            总票房=('box_office_ten_thousand', 'sum'),
            平均评分=('score', 'mean')
        ).reset_index()

        df_genre_plot = df_genre_stats[df_genre_stats['电影数量'] > 5]  # 过滤掉样本过少的类型
        if not df_genre_plot.empty:
            fig_roi = px.scatter(
                df_genre_plot, x='电影数量', y='平均票房', size='总票房', color='平均评分',
                text='genres_list', size_max=60, color_continuous_scale='Viridis',
                hover_name='genres_list',
                labels={'电影数量': '市场竞争度 (作品数)', '平均票房': '平均盈利能力 (均票房/万)'}
            )
            fig_roi.update_traces(textposition='top center')
            st.plotly_chart(fig_roi, use_container_width=True)

    # --- 选项卡3: 口碑与票房关系 (视觉升级) ---
    with tab3:
        df_plot = df_analysis.dropna(subset=['score', 'box_office_ten_thousand', 'year', 'comment_count'])

        st.header("动态演化散点图")
        st.info("播放动画，观看不同年份口碑-票房-热度的关系演化。")
        if not df_plot.empty and 'year' in df_plot.columns:
            # 确保年份是整数类型，以便动画正常播放
            df_plot['year'] = df_plot['year'].astype(int)
            fig_anim = px.scatter(
                df_plot.sort_values('year'),
                x='score', y='box_office_ten_thousand',
                animation_frame="year",
                animation_group="name_cn",
                size="comment_count",
                color="is_top250",
                hover_name="name_cn",
                log_y=True,
                size_max=55,
                labels={'score': '豆瓣评分', 'box_office_ten_thousand': '票房(万元)', 'comment_count': '评论数'}
            )
            st.plotly_chart(fig_anim, use_container_width=True)

        st.header("评分段票房分布脊线图")
        st.info("每一条“山脊”代表一个评分段的票房分布形态，比箱形图更直观。")
        df_ridge = df_plot.copy()
        df_ridge['score_segment'] = pd.cut(df_ridge['score'], bins=[0, 6, 7, 8, 9, 10.1],
                                           labels=['<6分', '6-7分', '7-8分', '8-9分', '9+分'], right=False)
        df_ridge.dropna(subset=['score_segment', 'box_office_ten_thousand'], inplace=True)

        if not df_ridge.empty:
            group_labels = df_ridge['score_segment'].cat.categories.tolist()
            hist_data = [df_ridge[df_ridge['score_segment'] == seg]['box_office_ten_thousand'].values for seg in
                         group_labels]

            # 创建分布图（近似脊线图）
            try:
                fig_ridge = create_distplot(hist_data, group_labels, show_hist=False, show_rug=False)
                fig_ridge.update_layout(title_text='评分段票房分布脊线图', yaxis_title="密度")
                st.plotly_chart(fig_ridge, use_container_width=True)
            except Exception as e:
                st.warning(f"无法生成脊线图，可能是数据分布问题。错误: {e}")