import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pymysql
import warnings
from math import pi

# 导入词云和中文分词库
import jieba
from wordcloud import WordCloud, STOPWORDS
from collections import Counter

# 因为不再进行票房预测，相关的机器学习库可以不导入或注释掉
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.preprocessing import StandardScaler


# --- Streamlit 页面配置 (必须是第一个 Streamlit 命令) ---
st.set_page_config(layout="wide", page_title="豆瓣电影分析 - Top 250")

# 设置 Matplotlib 字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或者 'Microsoft YaHei'，根据您的系统字体选择
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# --- 数据库连接 (使用 Streamlit 缓存资源，避免每次运行都重新连接) ---
@st.cache_resource
def get_database_connection_resource():
    """获取并缓存数据库连接"""
    try:
        # 请根据您的实际MySQL配置修改这些参数
        db = pymysql.connect(host='localhost',
                             user='root',
                             password='',  # <-- 请在这里填入您的MySQL密码
                             database='douban',  # 数据库名应为 'douban'
                             charset='utf8mb4')
        st.success("数据库连接成功！")
        return db
    except pymysql.err.OperationalError as e:
        st.error(f"数据库连接失败 (请检查MySQL服务或连接参数): {e}")
        st.stop()  # 停止应用运行，直到问题解决
    except Exception as e:
        st.error(f"发生未知数据库错误: {e}")
        st.stop()


db_connection = get_database_connection_resource()


# --- 数据加载与预处理 (使用 Streamlit 缓存数据) ---
@st.cache_data(ttl=3600)  # 数据缓存1小时
def load_and_preprocess_data(_db_conn) -> pd.DataFrame:
    """从数据库加载数据并进行预处理和特征工程"""
    st.info("正在加载和预处理豆瓣电影数据...")

    try:
        # 从 'movie' 表加载数据
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.read_sql(f'SELECT * FROM `movie`', _db_conn)
        st.success(f"成功从 'movie' 表加载 {len(df)} 行数据。")
        if df.empty:
            st.warning("`movie` 表中没有数据，请检查数据库或爬虫是否已运行。")
            return pd.DataFrame()  # 返回空DataFrame
    except Exception as e:
        st.error(f"从表 'movie' 加载数据失败: {e}")
        return pd.DataFrame()

    df = df.copy()  # 创建副本，避免SettingWithCopyWarning

    # --- 1. 列名映射 ---
    # 将数据库列名映射到代码中使用的更通用或更具描述性的列名
    column_mapping = {
        '中文名': '电影名',
        '评分': '豆瓣评分',
        '评价人数': '豆瓣评论数',
        '主演': '演员',  # 保持一致，方便后续处理
        '电影语录': '电影语录',  # 新增列
        '上映年份': '上映年份_str',  # 临时变量，用于后续转换
        '时长': '时长_str'  # 临时变量，用于后续转换
    }
    df.rename(columns=column_mapping, inplace=True)

    # --- 2. 上映年份处理 (从 varchar 提取四位数字年份) ---
    if '上映年份_str' in df.columns:
        # 尝试从字符串中提取四位数字作为年份
        df['上映年份'] = df['上映年份_str'].astype(str).str.extract(r'(\d{4})', expand=False)
        df['上映年份'] = pd.to_numeric(df['上映年份'], errors='coerce').fillna(-1).astype(int)
        df.drop(columns=['上映年份_str'], inplace=True)
    else:
        df['上映年份'] = -1
        st.warning("警告: 数据库中缺少列 '上映年份'。已填充默认值-1。")

    # --- 3. 时长处理 (从 varchar 提取分钟数) ---
    if '时长_str' in df.columns:
        # 尝试从字符串中提取数字，例如 "120分钟" -> 120
        df['时长'] = df['时长_str'].astype(str).str.extract(r'(\d+)', expand=False)
        df['时长'] = pd.to_numeric(df['时长'], errors='coerce').fillna(0).astype(int)
        df.drop(columns=['时长_str'], inplace=True)
    else:
        df['时长'] = 0
        st.warning("警告: 数据库中缺少列 '时长'。已填充默认值0。")

    # --- 4. 处理多值分类特征 ('导演', '演员', '类型', '地区', '语言') ---
    # 创建 '_list' 结尾的新列来存储列表形式的数据，这些列会保留用于EDA
    original_multi_value_cols = ['导演', '演员', '类型', '地区', '语言']
    for col in original_multi_value_cols:
        if col in df.columns:
            # 确保处理非字符串值，例如 NaN
            df[col + '_list'] = df[col].apply(
                lambda x: [item.strip() for item in str(x).split(',') if item.strip()] if pd.notnull(x) else []
            )
        else:
            df[col + '_list'] = [[]] * len(df)
            st.warning(f"警告: 数据库中缺少列 '{col}'。已创建空列表列。")

    # --- 5. 转换核心数值特征为正确的类型并处理缺失值 ---
    numeric_cols_to_convert = {
        '豆瓣评分': float,
        '豆瓣评论数': int,
        '时长': int
    }

    for col, dtype in numeric_cols_to_convert.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0.0 if dtype == float else 0)  # 填充NaN值
            try:
                df[col] = df[col].astype(dtype)
            except Exception as e:
                st.error(f"严重错误: 列 '{col}' 无法转换为 {dtype} 类型。请检查数据源中该列的值。")
                df[col] = 0.0 if dtype == float else 0  # 尝试安全地设为默认值
        else:
            df[col] = 0.0 if dtype == float else 0
            st.warning(f"警告: 数据库中缺少列 '{col}'。已创建并填充默认值。")

    # --- 6. 过滤不适合分析的电影 ---
    initial_rows = len(df)
    df = df[(df['豆瓣评分'] > 0) & (df['豆瓣评分'] <= 10)]  # 评分大于0且小于等于10
    df = df[df['豆瓣评论数'] > 0]  # 评论数大于0
    df = df[df['时长'] > 0]  # 时长大于0
    current_year = pd.Timestamp.now().year
    df = df[df['上映年份'].isin(range(1900, current_year + 2))]  # 限制年份范围，允许未来一年

    st.write(
        f"已过滤掉 {initial_rows - len(df)} 行不适合分析的电影（例如：评分0/评论数0/时长0/年份不合理）。剩余 {len(df)} 行。")

    if df.empty:
        st.warning("过滤后数据为空，无法进行分析。请检查数据源和过滤条件。")
        return pd.DataFrame()

    # --- 7. 清理原始字符串列 (保留_list版本用于EDA，删除原始字符串列) ---
    # original_multi_value_cols 已经用来生成_list列，现在可以删除原始列
    cols_to_drop_original_string = [col for col in original_multi_value_cols if col in df.columns]
    # '外文名', '电影语录', '详情URL' 这些原始列可能也需要保留，取决于实际需求。这里只删除原始的多值分类特征。
    df.drop(columns=cols_to_drop_original_string, inplace=True)

    # 豆瓣数据通常没有场均人数和票房，确保这些列不存在或为0
    if '票房(万元)' not in df.columns:
        df['票房(万元)'] = 0.0
    if '场均人数' not in df.columns:
        df['场均人数'] = 0.0

    st.success("数据预处理和特征工程完成。")
    return df


# --- 导演影响力雷达图辅助函数 ---
def plot_director_radar_chart(director_name, processed_df):
    """绘制导演影响力雷达图"""
    director_movies = processed_df[processed_df['导演_list'].apply(lambda x: director_name in x)]

    if director_movies.empty:
        st.warning(f"数据集中未找到导演 '{director_name}' 的电影信息。")
        return

    # 计算导演相关指标 (排除票房和场均人数，聚焦豆瓣评分、评论数、时长)
    total_movies = len(director_movies)
    avg_douban_score = director_movies['豆瓣评分'].mean()
    avg_length = director_movies['时长'].mean()
    avg_comments = director_movies['豆瓣评论数'].mean()

    # 计算整体平均指标，用于标准化
    overall_avg_douban_score = processed_df['豆瓣评分'].mean()
    overall_avg_length = processed_df['时长'].mean()
    overall_avg_comments = processed_df['豆瓣评论数'].mean()

    all_directors_flat = [d for sublist in processed_df['导演_list'] for d in sublist]
    unique_directors_count = len(pd.Series(all_directors_flat).unique())
    overall_total_movies_per_director = len(processed_df) / unique_directors_count if unique_directors_count > 0 else 1

    categories = ['电影数量', '平均评分', '平均片长', '平均评论数']
    num_vars = len(categories)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]  # 角度计算
    angles += angles[:1]  # 闭合雷达图

    # 标准化指标值，避免某个值过大或过小导致雷达图失衡
    epsilon = 1e-6  # 避免除以零
    values_director = [
        total_movies / (overall_total_movies_per_director + epsilon),
        avg_douban_score / (overall_avg_douban_score + epsilon),
        avg_length / (overall_avg_length + epsilon),
        avg_comments / (overall_avg_comments + epsilon)
    ]
    values_director = [min(v, 2.0) for v in values_director]  # 将标准化后的值限制在一个合理范围内
    values_director += values_director[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, values_director, color='red', alpha=0.25)
    ax.plot(angles, values_director, color='red', linewidth=2, linestyle='solid', label=director_name)

    ax.set_yticklabels([])  # 不显示径向刻度标签
    ax.set_xticks(angles[:-1])  # 设置刻度位置
    ax.set_xticklabels(categories, color='grey', size=12)  # 设置刻度标签

    # 添加具体数值标签
    for i, (angle, value) in enumerate(zip(angles[:-1], values_director[:-1])):
        text_val = ""
        if categories[i] == '电影数量':
            text_val = f'{total_movies:.0f}部'
        elif categories[i] == '平均评分':
            text_val = f'{avg_douban_score:.1f}分'
        elif categories[i] == '平均片长':
            text_val = f'{avg_length:.0f}分钟'
        elif categories[i] == '平均评论数':
            text_val = f'{avg_comments:.0f}'  # 显示原始平均评论数，不转为万

        if value * 1.15 > ax.get_rmax():
            ax.set_rmax(value * 1.2)

        ax.text(angle, value * 1.15, text_val, color='black', size=10,
                horizontalalignment='center' if angle % (pi / 2) == 0 else ('left' if 0 < angle < pi else 'right'),
                verticalalignment='center' if angle == 0 else ('bottom' if 0 < angle < pi else 'top'))

    ax.set_title(f'导演 {director_name} 影响力雷达图 (相对于平均水平)', va='bottom', fontsize=16)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    st.pyplot(fig)


# --- 词云生成辅助函数 ---
def generate_wordcloud(text_corpus, title="词云", max_words=200, stopwords=None):
    if not text_corpus.strip():
        st.warning(f"{title} - 语料为空，无法生成词云。")
        return

    # 添加一些常见的中文停用词，以及可能来自电影语录的通用词
    if stopwords is None:
        stopwords = set(STOPWORDS)
        custom_stopwords = {
            '的', '是', '了', '和', '也', '在', '我', '你', '他', '她', '它',
            '这部', '一个', '一种', '没有', '就是', '我们', '他们', '她们', '一个', '一些',
            '什么', '如此', '电影', '故事', '人生', '时间', '世界', '所有', '一个', '一场',
            '一段', '生活', '爱', '一切', '自己', '永远', '可以', '为了', '如果', '这部电影'
        }
        stopwords.update(custom_stopwords)

    # 中文分词
    seg_list = jieba.cut(text_corpus, cut_all=False)  # 精确模式
    filtered_words = [word for word in seg_list if len(word) > 1 and word not in stopwords]

    if not filtered_words:
        st.warning(f"{title} - 分词后没有有效词汇，无法生成词云。")
        return

    text = " ".join(filtered_words)

    wc = WordCloud(
        font_path='FZSTK.TTF',  # 确保这里指向您的中文TrueType字体文件
        background_color="white",
        max_words=max_words,
        width=1000,
        height=600,
        margin=2,
        random_state=42,
        collocations=False,  # 不包含词组
        stopwords=stopwords
    )
    wc.generate(text)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis("off")
    ax.set_title(title, fontsize=18)
    st.pyplot(fig)


# --- 主应用程序逻辑 ---
st.title("豆瓣电影分析 - Top 250 🎬")
st.markdown("---")

st.sidebar.header("导航")
# 移除票房预测页面
page = st.sidebar.radio("选择页面", ["数据概览", "探索性分析", "导演影响力分析", "电影语录词云"])

# 加载数据 (在所有页面逻辑之前加载一次)
processed_df = load_and_preprocess_data(db_connection)

if processed_df.empty:
    st.error("数据加载或预处理失败，请检查数据库和数据。")
    st.stop()  # 如果数据为空，则停止应用，避免后续报错

# === 页面1: 数据概览 ===
if page == "数据概览":
    st.header("📊 数据概览")

    # 关键指标展示
    col1, col2, col3 = st.columns(3)  # 只有3列指标
    with col1:
        st.metric("电影总数", len(processed_df))
    with col2:
        st.metric("平均评分", f"{processed_df['豆瓣评分'].mean():.2f}")
    with col3:
        min_year = processed_df['上映年份'].min() if not processed_df['上映年份'].empty else 'N/A'
        max_year = processed_df['上映年份'].max() if not processed_df['上映年份'].empty else 'N/A'
        st.metric("年份范围", f"{min_year}-{max_year}")

    st.subheader("数据表预览")
    # 显示部分核心列
    display_cols = ['电影名', '豆瓣评分', '豆瓣评论数', '上映年份', '时长', '导演_list', '类型_list', '电影语录']
    available_cols = [col for col in display_cols if col in processed_df.columns]
    st.dataframe(processed_df[available_cols].head(10))

    st.subheader("数据分布统计")
    numeric_cols = ['豆瓣评分', '豆瓣评论数', '时长']  # 移除票房、场均人数
    available_numeric = [col for col in numeric_cols if col in processed_df.columns]
    if available_numeric:
        st.dataframe(processed_df[available_numeric].describe().transpose())
    else:
        st.warning("无可用的数值列进行统计描述。")

# === 页面2: 探索性分析 ===
elif page == "探索性分析":
    st.header("🔍 探索性数据分析")

    # 评分分布
    st.subheader("电影评分分布分析")
    col1, col2 = st.columns(2)

    with col1:
        if '豆瓣评分' in processed_df.columns:
            fig_hist = px.histogram(processed_df, x='豆瓣评分', nbins=20,
                                    title='豆瓣评分分布直方图',
                                    labels={'豆瓣评分': '豆瓣评分', 'count': '电影数量'})
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.warning("豆瓣评分数据不可用。")

    with col2:
        if '豆瓣评论数' in processed_df.columns and '豆瓣评分' in processed_df.columns:
            # 评分 vs 评论数散点图
            fig_scatter = px.scatter(processed_df, x='豆瓣评论数', y='豆瓣评分',
                                     hover_data=['电影名'] if '电影名' in processed_df.columns else None,
                                     title='豆瓣评论数 vs 豆瓣评分',
                                     log_x=True)  # 评论数可能分布很广，用对数轴更清晰
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("豆瓣评论数或豆瓣评分数据不可用。")

    # 年度电影数量和平均评分趋势
    st.subheader("年度电影数量与平均评分趋势")
    if '上映年份' in processed_df.columns and '豆瓣评分' in processed_df.columns:
        yearly_stats = processed_df.groupby('上映年份').agg(
            电影数量=('电影名', 'count') if '电影名' in processed_df.columns else ('豆瓣评分', 'count'),
            平均评分=('豆瓣评分', 'mean')
        ).reset_index()
        yearly_stats = yearly_stats[yearly_stats['上映年份'] > 1900]  # 过滤不合理的年份

        if not yearly_stats.empty:
            fig_yearly = make_subplots(
                rows=1, cols=2,
                subplot_titles=('年度电影数量', '年度平均评分'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )

            fig_yearly.add_trace(go.Scatter(x=yearly_stats['上映年份'], y=yearly_stats['电影数量'],
                                            name='电影数量', line=dict(color='red')), row=1, col=1)
            fig_yearly.add_trace(go.Scatter(x=yearly_stats['上映年份'], y=yearly_stats['平均评分'],
                                            name='平均评分', line=dict(color='purple')), row=1, col=2)

            fig_yearly.update_layout(height=400, showlegend=False, title_text="年度电影市场趋势分析")
            st.plotly_chart(fig_yearly, use_container_width=True)
        else:
            st.warning("年度趋势数据不足以绘制图表。")
    else:
        st.warning("年度趋势分析所需数据（上映年份、评分）不可用。")

    # 类型分析
    st.subheader("电影类型分析")
    if '类型_list' in processed_df.columns and '豆瓣评分' in processed_df.columns:
        all_genres = [genre for sublist in processed_df['类型_list'] for genre in sublist]
        genre_counts = pd.Series(all_genres).value_counts().head(15)

        col1, col2 = st.columns(2)
        with col1:
            fig_genre_bar = px.bar(x=genre_counts.values, y=genre_counts.index,
                                   orientation='h', title='电影类型分布',
                                   labels={'x': '电影数量', 'y': '类型'})
            st.plotly_chart(fig_genre_bar, use_container_width=True)

        with col2:
            # 类型平均评分
            genre_scores = {}
            for idx, genres in processed_df['类型_list'].items():
                score = processed_df.loc[idx, '豆瓣评分']
                for genre in genres:
                    if genre not in genre_scores:
                        genre_scores[genre] = []
                    genre_scores[genre].append(score)

            MIN_MOVIES_FOR_AVG = 5  # 至少5部电影才能计算平均分
            genre_avg_score = {k: np.mean(v) for k, v in genre_scores.items() if len(v) >= MIN_MOVIES_FOR_AVG}
            genre_avg_score = dict(sorted(genre_avg_score.items(), key=lambda x: x[1], reverse=True)[:10])

            if genre_avg_score:
                fig_genre_avg = px.bar(x=list(genre_avg_score.values()), y=list(genre_avg_score.keys()),
                                       orientation='h', title=f'各类型平均评分Top10 (至少{MIN_MOVIES_FOR_AVG}部电影)',
                                       labels={'x': '平均评分', 'y': '类型'})
                st.plotly_chart(fig_genre_avg, use_container_width=True)
            else:
                st.info("没有足够数据计算类型平均评分。")
    else:
        st.warning("类型数据或豆瓣评分数据不可用。")

    # 电影时长分布
    st.subheader("电影时长分布")
    if '时长' in processed_df.columns:
        fig_duration = px.histogram(processed_df, x='时长', nbins=30,
                                    title='电影时长分布直方图',
                                    labels={'时长': '时长(分钟)', 'count': '电影数量'})
        st.plotly_chart(fig_duration, use_container_width=True)
    else:
        st.warning("电影时长数据不可用。")


# === 页面3: 导演影响力分析 ===
elif page == "导演影响力分析":
    st.header("🎬 导演影响力分析")

    if '导演_list' in processed_df.columns and '豆瓣评分' in processed_df.columns and '豆瓣评论数' in processed_df.columns:
        all_directors = [d for sublist in processed_df['导演_list'] for d in sublist]
        director_stats = {}

        for director in set(all_directors):
            director_movies = processed_df[processed_df['导演_list'].apply(lambda x: director in x)]
            if len(director_movies) >= 2:  # 至少2部电影，数据更有意义
                director_stats[director] = {
                    '电影数量': len(director_movies),
                    '平均评分': director_movies['豆瓣评分'].mean(),
                    '平均时长': director_movies['时长'].mean(),
                    '总评论数': director_movies['豆瓣评论数'].sum()
                }

        director_df = pd.DataFrame(director_stats).T.reset_index()
        director_df.columns = ['导演', '电影数量', '平均评分', '平均时长', '总评论数']
        # 排序以总评论数或电影数量
        director_df = director_df.sort_values('总评论数', ascending=False)

        if not director_df.empty:
            # 顶级导演排行榜
            st.subheader("📊 导演排行榜")

            tab1, tab2, tab3 = st.tabs(["电影数量排行", "平均评分排行", "总评论数排行"])

            with tab1:
                top_directors_count = director_df.sort_values('电影数量', ascending=False).head(15)
                fig_director_count = px.bar(top_directors_count, x='电影数量', y='导演',
                                            orientation='h', title='导演电影数量Top15',
                                            labels={'电影数量': '电影数量', '导演': '导演'})
                st.plotly_chart(fig_director_count, use_container_width=True)

            with tab2:
                # 过滤出至少3部电影的导演，使得平均评分更具代表性
                top_directors_avg_score = director_df[director_df['电影数量'] >= 3].sort_values('平均评分',
                                                                                                ascending=False).head(
                    15)
                fig_director_avg_score = px.bar(top_directors_avg_score, x='平均评分', y='导演',
                                                orientation='h', title='导演平均评分Top15 (至少3部电影)',
                                                labels={'平均评分': '平均评分', '导演': '导演'})
                st.plotly_chart(fig_director_avg_score, use_container_width=True)

            with tab3:
                top_directors_comments = director_df.sort_values('总评论数', ascending=False).head(15)
                fig_director_comments = px.bar(top_directors_comments, x='总评论数', y='导演',
                                               orientation='h', title='导演总评论数Top15',
                                               labels={'总评论数': '总评论数', '导演': '导演'})
                st.plotly_chart(fig_director_comments, use_container_width=True)

            # 导演详细分析
            st.subheader("🔍 导演详细分析")

            # 导演选择器
            available_directors = sorted(director_df['导演'].tolist())
            selected_director = st.selectbox("选择导演进行详细分析:", available_directors)

            if selected_director:
                director_movies = processed_df[processed_df['导演_list'].apply(lambda x: selected_director in x)]

                # 导演基本信息
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("电影数量", len(director_movies))
                with col2:
                    st.metric("平均评分", f"{director_movies['豆瓣评分'].mean():.1f}")
                with col3:
                    st.metric("总评论数", f"{director_movies['豆瓣评论数'].sum():,.0f}")

                # 导演电影列表
                st.subheader(f"{selected_director} 的电影作品")
                movie_cols = ['电影名', '豆瓣评分', '豆瓣评论数', '上映年份', '时长']
                available_movie_cols = [col for col in movie_cols if col in director_movies.columns]

                display_movies = director_movies[available_movie_cols].sort_values('豆瓣评分', ascending=False)
                st.dataframe(display_movies, use_container_width=True)

                # 导演作品趋势分析
                if len(director_movies) >= 3:  # 至少3部电影才能看趋势
                    st.subheader(f"{selected_director} 作品评分趋势")

                    director_movies_sorted = director_movies.sort_values('上映年份')

                    fig_trend = go.Figure()
                    fig_trend.add_trace(go.Scatter(x=director_movies_sorted['上映年份'],
                                                   y=director_movies_sorted['豆瓣评分'],
                                                   mode='lines+markers',
                                                   name='豆瓣评分',
                                                   line=dict(color='orange')))

                    fig_trend.update_layout(height=400, title_text=f"{selected_director} 作品平均评分趋势",
                                            showlegend=False)
                    st.plotly_chart(fig_trend, use_container_width=True)
                else:
                    st.info(f"导演 {selected_director} 作品数量不足，无法绘制趋势图。")

                # 雷达图分析
                st.subheader(f"{selected_director} 影响力雷达图")
                plot_director_radar_chart(selected_director, processed_df)

            # 导演对比分析
            st.subheader("🆚 导演对比分析")

            col1, col2 = st.columns(2)
            with col1:
                director1 = st.selectbox("选择导演1:", available_directors, key="director1")
            with col2:
                director2 = st.selectbox("选择导演2:", available_directors, key="director2",
                                         index=1 if len(available_directors) > 1 else 0)

            if director1 and director2 and director1 != director2:
                director1_movies = processed_df[processed_df['导演_list'].apply(lambda x: director1 in x)]
                director2_movies = processed_df[processed_df['导演_list'].apply(lambda x: director2 in x)]

                comparison_df = pd.DataFrame({
                    '指标': ['电影数量', '平均评分', '平均时长(分钟)', '总评论数'],
                    director1: [
                        len(director1_movies),
                        director1_movies['豆瓣评分'].mean(),
                        director1_movies['时长'].mean(),
                        director1_movies['豆瓣评论数'].sum()
                    ],
                    director2: [
                        len(director2_movies),
                        director2_movies['豆瓣评分'].mean(),
                        director2_movies['时长'].mean(),
                        director2_movies['豆瓣评论数'].sum()
                    ]
                })

                st.dataframe(comparison_df.round(2), use_container_width=True)
            elif director1 == director2 and director1:
                st.warning("请选择两位不同的导演进行对比。")
            else:
                st.info("请选择两位导演进行对比。")

        else:
            st.warning("没有足够的导演数据进行分析。")
    else:
        st.error("导演相关数据（导演列表、评分、评论数）不可用，请检查数据预处理。")

# === 页面4: 电影语录词云 ===
elif page == "电影语录词云":
    st.header("☁️ 电影语录词云分析")

    if '电影语录' in processed_df.columns:
        all_quotes = " ".join(processed_df['电影语录'].dropna().astype(str).tolist())

        st.markdown("### 豆瓣 Top 250 电影语录总览词云")
        generate_wordcloud(all_quotes, title="豆瓣 Top 250 电影语录词云")

        st.markdown("---")
        st.subheader("按电影类型生成词云")
        if '类型_list' in processed_df.columns:
            all_types = [genre for sublist in processed_df['类型_list'] for genre in sublist]
            unique_types = sorted(list(set(all_types)))

            if unique_types:
                selected_type = st.selectbox("选择电影类型：", ['所有类型'] + unique_types)

                if selected_type == '所有类型':
                    generate_wordcloud(all_quotes, title="所有电影类型语录词云")
                else:
                    type_quotes = processed_df[processed_df['类型_list'].apply(lambda x: selected_type in x)][
                        '电影语录'].dropna().astype(str).tolist()
                    if type_quotes:
                        generate_wordcloud(" ".join(type_quotes), title=f"{selected_type} 电影语录词云")
                    else:
                        st.info(f"类型 '{selected_type}' 没有电影语录数据。")
            else:
                st.warning("没有可用的电影类型数据。")
        else:
            st.warning("电影类型数据不可用。")

    else:
        st.warning("数据库中没有 '电影语录' 列，无法生成词云。")

# 侧边栏额外信息
st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 数据洞察")

if not processed_df.empty:
    # 显示一些有趣的统计
    highest_rated_col = '豆瓣评分'
    movie_name_col = '电影名'

    if highest_rated_col in processed_df.columns and not processed_df[highest_rated_col].empty:
        highest_rated = processed_df.loc[processed_df[highest_rated_col].idxmax()]
        st.sidebar.markdown(f"**最高评分电影:**")
        if movie_name_col in processed_df.columns:
            st.sidebar.write(f"🎬 {highest_rated.get(movie_name_col, 'N/A')}")
        st.sidebar.write(f"⭐ {highest_rated[highest_rated_col]:.1f}分")
    else:
        st.sidebar.info("最高评分电影数据不可用。")

    # 显示数据更新时间
    st.sidebar.markdown("---")
    st.sidebar.markdown("**数据说明:**")
    st.sidebar.write("• 数据来源: 豆瓣电影 (Top 250)")
    st.sidebar.write("• 包含评分、导演、类型、语录等信息")
    st.sidebar.write("• 专注于电影特性和影响力分析")
else:
    st.sidebar.warning("暂无数据可供分析。")

# 页脚
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🎬 豆瓣电影分析系统 | 基于 Top 250 数据的洞察 📊</p>
        <p><small>数据驱动决策，助力电影爱好者发现好片</small></p>
    </div>
    """,
    unsafe_allow_html=True
)