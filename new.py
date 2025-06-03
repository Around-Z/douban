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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
# import jieba # 暂时未使用，可按需保留
# from wordcloud import WordCloud # 暂时未使用，可按需保留
from collections import Counter

# --- Streamlit 页面配置 (必须是第一个 Streamlit 命令) ---
st.set_page_config(layout="wide", page_title="电影票房分析与预测")

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
                             password='',
                             database='douban',
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
    st.info("正在加载和预处理电影数据...")

    try:
        # 忽略SQLalchemy的警告，因为我们直接用pymysql
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.read_sql(f'SELECT * FROM `maoyan_movie`', _db_conn)
        st.success(f"成功从 'maoyan_movie' 表加载 {len(df)} 行数据。")
        if df.empty:
            st.warning("maoyan_movie 表中没有数据，请先运行爬虫填充数据。")
            return pd.DataFrame()  # 返回空DataFrame
    except Exception as e:
        st.error(f"从表 'maoyan_movie' 加载数据失败: {e}")
        return pd.DataFrame()

    df = df.copy()  # 创建副本，避免SettingWithCopyWarning

    # 1. 上映日期处理
    df['上映日期'] = pd.to_datetime(df['上映日期'], errors='coerce')
    if df['上映日期'].isnull().any():
        st.warning("警告: '上映日期' 列中存在无法解析的日期，已转换为 NaT。")

    # 提取时间特征
    df['上映年份'] = df['上映日期'].dt.year.fillna(-1).astype(int)
    df['上映月份'] = df['上映日期'].dt.month.fillna(-1).astype(int)
    df['上映日'] = df['上映日期'].dt.day.fillna(-1).astype(int)
    df['上映星期几'] = df['上映日期'].dt.dayofweek.fillna(-1).astype(int)  # 0=星期一, 6=星期日
    df['是否周末'] = df['上映星期几'].isin([5, 6]).astype(int)  # 5:周六, 6:周日

    # 2. 处理多值分类特征 ('导演', '演员', '类型', '地区', '语言')
    # 创建 '_list' 结尾的新列来存储列表形式的数据，这些列会保留用于EDA
    original_multi_value_cols = ['导演', '演员', '类型', '地区', '语言']
    for col in original_multi_value_cols:
        if col in df.columns:  # 确保列存在
            df[col + '_list'] = df[col].apply(
                lambda x: [item.strip() for item in str(x).split(',') if item.strip()] if pd.notnull(x) else []
            )
        else:
            df[col + '_list'] = [[]] * len(df)  # 如果列不存在，则创建空列表列
            st.warning(f"警告: 数据库中缺少列 '{col}'。已创建空列表列。")

    # --- 3. 转换核心数值特征为正确的类型并处理缺失值 ---
    numeric_cols_to_convert = {
        '豆瓣评分': float,
        '豆瓣评论数': int,
        '时长': int,
        '票房(万元)': float,
        '场均人数': float  # 假设 '场均人数' 列存在
    }

    for col, dtype in numeric_cols_to_convert.items():
        if col in df.columns:
            # 尝试将列转换为数字类型，无法转换的变为NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # 填充NaN值：浮点数用0.0，整数用0
            df[col] = df[col].fillna(0.0 if dtype == float else 0)
            # 再次强制转换为目标类型
            try:
                df[col] = df[col].astype(dtype)
            except Exception as e:
                st.error(f"严重错误: 列 '{col}' 无法转换为 {dtype} 类型。请检查数据源中该列的非数字或列表值。")
                # 如果发生错误，将该列设为全零以避免后续崩溃，但这表示数据质量问题
                df[col] = 0.0 if dtype == float else 0
        else:
            df[col] = 0.0 if dtype == float else 0  # 如果列不存在，则创建并填充默认值
            st.warning(f"警告: 数据库中缺少列 '{col}'。已创建并填充默认值。")

    # 过滤掉不适合分析或预测的电影
    initial_rows = len(df)
    # 票房为0或NaN的过滤
    df = df[df['票房(万元)'] > 0]
    # 豆瓣评分大于0且小于等于10
    df = df[(df['豆瓣评分'] > 0) & (df['豆瓣评分'] <= 10)]
    # 评论数大于0
    df = df[df['豆瓣评论数'] > 0]
    # 时长大于0
    df = df[df['时长'] > 0]
    # 过滤掉未来上映的电影，或者上映年份明显不合理（比如-1）的
    current_year = pd.Timestamp.now().year
    df = df[df['上映年份'].isin(range(1900, current_year + 1))]  # 限制年份范围

    st.write(
        f"已过滤掉 {initial_rows - len(df)} 行不适合分析或预测的电影（例如：未来上映/票房0/评分0/评论数0/时长0）。剩余 {len(df)} 行。")

    if df.empty:
        st.warning("过滤后数据为空，无法进行分析。请检查数据源和过滤条件。")
        return pd.DataFrame()

    # --- 4. 对选定特征进行 One-Hot 编码 ---
    # 统计并选择Top N，避免生成过多特征
    TOP_N_DIRECTORS = 30
    TOP_N_TYPES = 15
    TOP_N_PLACES = 10
    TOP_N_LANGS = 10

    all_directors = [d for sublist in df['导演_list'] for d in sublist]
    all_types = [t for sublist in df['类型_list'] for t in sublist]
    all_places = [p for sublist in df['地区_list'] for p in sublist]
    all_langs = [l for sublist in df['语言_list'] for l in sublist]

    top_directors = pd.Series(all_directors).value_counts().head(TOP_N_DIRECTORS).index.tolist()
    top_types = pd.Series(all_types).value_counts().head(TOP_N_TYPES).index.tolist()
    top_places = pd.Series(all_places).value_counts().head(TOP_N_PLACES).index.tolist()
    top_langs = pd.Series(all_langs).value_counts().head(TOP_N_LANGS).index.tolist()

    def one_hot_encode_multi_value(df_col_list, top_items_list, prefix):
        """对多值特征进行One-Hot编码"""
        # 创建一个空DataFrame，列名为 'prefix_item'
        temp_df = pd.DataFrame(0, index=df_col_list.index, columns=[f"{prefix}_{item}" for item in top_items_list])
        for idx, items in df_col_list.items():
            for item in items:
                col_name = f"{prefix}_{item}"
                if col_name in temp_df.columns:
                    temp_df.loc[idx, col_name] = 1
        return temp_df

    st.write("正在进行 One-Hot 编码...")
    df_directors_ohe = one_hot_encode_multi_value(df['导演_list'], top_directors, '导演')
    df_types_ohe = one_hot_encode_multi_value(df['类型_list'], top_types, '类型')
    df_places_ohe = one_hot_encode_multi_value(df['地区_list'], top_places, '地区')
    df_langs_ohe = one_hot_encode_multi_value(df['语言_list'], top_langs, '语言')

    # 将编码后的 DataFrame 合并回主 DataFrame
    df = pd.concat([df, df_directors_ohe, df_types_ohe, df_places_ohe, df_langs_ohe], axis=1)

    # --- 5. 清理原始字符串列 (保留_list版本用于EDA，删除原始字符串列) ---
    cols_to_drop_original_string = [col for col in original_multi_value_cols if col in df.columns]
    df.drop(columns=cols_to_drop_original_string, inplace=True)

    st.success("数据预处理和特征工程完成。")
    return df


# --- 导演影响力雷达图辅助函数 ---
def plot_director_radar_chart(director_name, processed_df):
    """绘制导演影响力雷达图"""
    director_movies = processed_df[processed_df['导演_list'].apply(lambda x: director_name in x)]

    if director_movies.empty:
        st.warning(f"数据集中未找到导演 '{director_name}' 的电影信息。")
        return

    # 计算导演相关指标
    total_movies = len(director_movies)
    avg_box_office = director_movies['票房(万元)'].mean()
    avg_douban_score = director_movies['豆瓣评分'].mean()
    avg_length = director_movies['时长'].mean()
    avg_avg_people = director_movies['场均人数'].mean()  # 假设 '场均人数' 列存在

    # 计算整体平均指标，用于标准化
    overall_avg_box_office = processed_df['票房(万元)'].mean()
    overall_avg_douban_score = processed_df['豆瓣评分'].mean()
    overall_avg_length = processed_df['时长'].mean()
    overall_avg_avg_people = processed_df['场均人数'].mean()

    all_directors_flat = [d for sublist in processed_df['导演_list'] for d in sublist]
    unique_directors_count = len(pd.Series(all_directors_flat).unique())
    # 考虑整体导演平均电影数量，避免除以零
    overall_total_movies_per_director = len(processed_df) / unique_directors_count if unique_directors_count > 0 else 1

    categories = ['电影数量', '平均票房', '平均评分', '平均片长', '场均人数']
    num_vars = len(categories)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]  # 角度计算
    angles += angles[:1]  # 闭合雷达图

    # 标准化指标值，避免某个值过大或过小导致雷达图失衡
    epsilon = 1e-6  # 避免除以零
    values_director = [
        total_movies / (overall_total_movies_per_director + epsilon),
        avg_box_office / (overall_avg_box_office + epsilon),
        avg_douban_score / (overall_avg_douban_score + epsilon),
        avg_length / (overall_avg_length + epsilon),
        avg_avg_people / (overall_avg_avg_people + epsilon)
    ]
    # 将标准化后的值限制在一个合理范围内，例如不超过2倍平均值，增强可读性
    values_director = [min(v, 2.0) for v in values_director]
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
        elif categories[i] == '平均票房':
            text_val = f'{avg_box_office / 10000:.1f}亿'
        elif categories[i] == '平均评分':
            text_val = f'{avg_douban_score:.1f}分'
        elif categories[i] == '平均片长':
            text_val = f'{avg_length:.0f}分钟'
        elif categories[i] == '场均人数':
            text_val = f'{avg_avg_people:.1f}人'

        # 动态调整最大半径以容纳标签
        if value * 1.15 > ax.get_rmax():
            ax.set_rmax(value * 1.2)  # 增加一些裕量

        ax.text(angle, value * 1.15, text_val, color='black', size=10,
                horizontalalignment='center' if angle % (pi / 2) == 0 else ('left' if 0 < angle < pi else 'right'),
                verticalalignment='center' if angle == 0 else ('bottom' if 0 < angle < pi else 'top'))

    ax.set_title(f'导演 {director_name} 影响力雷达图 (相对于平均水平)', va='bottom', fontsize=16)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    st.pyplot(fig)


# --- 票房预测模型 ---
@st.cache_data(ttl=3600)
def build_prediction_model(processed_df):
    """构建票房预测模型 (随机森林和线性回归)"""
    st.info("正在构建票房预测模型...")

    # 选择特征列
    feature_cols = []

    # 数值特征
    numeric_features = ['豆瓣评分', '豆瓣评论数', '时长', '上映年份', '上映月份', '是否周末', '场均人数']
    feature_cols.extend([col for col in numeric_features if col in processed_df.columns])

    # One-Hot编码特征 (根据数据预处理中生成的列名动态获取)
    ohe_cols = [col for col in processed_df.columns if
                any(col.startswith(prefix) for prefix in ['导演_', '类型_', '地区_', '语言_'])]
    feature_cols.extend(ohe_cols)

    # 过滤掉DataFrame中实际不存在的特征列，并检查是否含有非标量数据
    final_feature_cols = []
    for col in feature_cols:
        if col in processed_df.columns:
            # 检查列是否包含列表、元组或字典等非标量值
            if processed_df[col].apply(lambda x: isinstance(x, (list, tuple, dict))).any():
                st.warning(f"特征列 '{col}' 包含非标量值（列表/元组/字典）。已跳过该列进行模型训练。")
            else:
                final_feature_cols.append(col)
        else:
            st.warning(f"特征列 '{col}' 在处理后的数据中不存在。已跳过该列。")

    if not final_feature_cols:
        st.error("没有可用于预测的有效特征列，请检查数据预处理步骤。")
        return None

    # 准备训练数据
    X = processed_df[final_feature_cols].fillna(0)  # 填充可能存在的NaN值
    y = processed_df['票房(万元)']

    if X.empty or y.empty:
        st.error("训练数据为空，无法构建模型。请检查数据源和过滤条件。")
        return None

    # 再次确认X中没有非数值类型，特别是object类型
    non_numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    if len(non_numeric_cols) < len(X.columns):
        st.error(
            f"特征数据X中包含非数值列，这可能导致StandardScaler失败。非数值列: {list(set(X.columns) - set(non_numeric_cols))}")
        st.stop()  # 停止运行，要求用户检查数据

    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 特征标准化 (对数值特征进行标准化)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练模型
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)  # 使用所有CPU核心
    lr_model = LinearRegression(n_jobs=-1)

    rf_model.fit(X_train_scaled, y_train)
    lr_model.fit(X_train_scaled, y_train)

    # 预测
    rf_pred = rf_model.predict(X_test_scaled)
    lr_pred = lr_model.predict(X_test_scaled)

    # 评估
    rf_r2 = r2_score(y_test, rf_pred)
    lr_r2 = r2_score(y_test, lr_pred)
    rf_mse = mean_squared_error(y_test, rf_pred)
    lr_mse = mean_squared_error(y_test, lr_pred)

    st.success("模型训练完成！")
    return {
        'rf_model': rf_model,
        'lr_model': lr_model,
        'scaler': scaler,
        'feature_cols': final_feature_cols,  # 使用实际用于训练的特征列
        'rf_r2': rf_r2,
        'lr_r2': lr_r2,
        'rf_mse': rf_mse,
        'lr_mse': lr_mse,
        'y_test': y_test,
        'rf_pred': rf_pred,
        'lr_pred': lr_pred
    }


# --- 主应用程序逻辑 ---
st.title("电影票房预测与观众偏好分析 🎬")
st.markdown("---")

st.sidebar.header("导航")
page = st.sidebar.radio("选择页面", ["数据概览", "探索性分析", "票房预测与建模", "导演影响力分析"])

# 加载数据 (在所有页面逻辑之前加载一次)
processed_df = load_and_preprocess_data(db_connection)

if processed_df.empty:
    st.error("数据加载或预处理失败，请检查数据库和数据。")
    st.stop()  # 如果数据为空，则停止应用，避免后续报错

# === 页面1: 数据概览 ===
if page == "数据概览":
    st.header("📊 数据概览")

    # 关键指标展示
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("电影总数", len(processed_df))
    with col2:
        st.metric("平均票房", f"{processed_df['票房(万元)'].mean() / 10000:.2f}亿")
    with col3:
        st.metric("平均评分", f"{processed_df['豆瓣评分'].mean():.2f}")
    with col4:
        min_year = processed_df['上映年份'].min() if not processed_df['上映年份'].empty else 'N/A'
        max_year = processed_df['上映年份'].max() if not processed_df['上映年份'].empty else 'N/A'
        st.metric("年份范围", f"{min_year}-{max_year}")

    st.subheader("数据表预览")
    # 显示部分核心列
    display_cols = ['电影名', '票房(万元)', '豆瓣评分', '上映日期', '时长', '场均人数', '导演_list', '类型_list']
    available_cols = [col for col in display_cols if col in processed_df.columns]
    st.dataframe(processed_df[available_cols].head(10))

    st.subheader("数据分布统计")
    numeric_cols = ['票房(万元)', '豆瓣评分', '豆瓣评论数', '时长', '场均人数']
    available_numeric = [col for col in numeric_cols if col in processed_df.columns]
    if available_numeric:
        st.dataframe(processed_df[available_numeric].describe().transpose())
    else:
        st.warning("无可用的数值列进行统计描述。")

# === 页面2: 探索性分析 ===
elif page == "探索性分析":
    st.header("🔍 探索性数据分析")

    # 票房分布
    st.subheader("票房分布分析")
    col1, col2 = st.columns(2)

    with col1:
        if '票房(万元)' in processed_df.columns:
            fig_hist = px.histogram(processed_df, x='票房(万元)', nbins=50,
                                    title='票房分布直方图',
                                    labels={'票房(万元)': '票房(万元)', 'count': '电影数量'})
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.warning("票房数据不可用。")

    with col2:
        if '豆瓣评分' in processed_df.columns and '票房(万元)' in processed_df.columns:
            # 票房vs评分散点图
            # 由于安装了statsmodels，trendline="ols"现在应该可以正常工作
            fig_scatter = px.scatter(processed_df, x='豆瓣评分', y='票房(万元)',
                                     hover_data=['电影名'] if '电影名' in processed_df.columns else None,
                                     title='票房 vs 豆瓣评分',
                                     trendline="ols")  # 添加OLS趋势线
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("豆瓣评分或票房数据不可用。")

    # 年度票房趋势
    st.subheader("年度电影市场趋势")
    if '上映年份' in processed_df.columns and '票房(万元)' in processed_df.columns and '豆瓣评分' in processed_df.columns:
        yearly_stats = processed_df.groupby('上映年份').agg(
            总票房=('票房(万元)', 'sum'),
            平均票房=('票房(万元)', 'mean'),
            电影数量=('电影名', 'count') if '电影名' in processed_df.columns else ('票房(万元)', 'count'),
            平均评分=('豆瓣评分', 'mean')
        ).reset_index()
        # 过滤掉不合理的年份（如-1）
        yearly_stats = yearly_stats[yearly_stats['上映年份'] > 1900]

        if not yearly_stats.empty:
            fig_yearly = make_subplots(
                rows=2, cols=2,
                subplot_titles=('年度总票房', '年度平均票房', '年度电影数量', '年度平均评分'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )

            fig_yearly.add_trace(go.Scatter(x=yearly_stats['上映年份'], y=yearly_stats['总票房'] / 10000,
                                            name='总票房(亿)', line=dict(color='blue')), row=1, col=1)
            fig_yearly.add_trace(go.Scatter(x=yearly_stats['上映年份'], y=yearly_stats['平均票房'] / 10000,
                                            name='平均票房(亿)', line=dict(color='green')), row=1, col=2)
            fig_yearly.add_trace(go.Scatter(x=yearly_stats['上映年份'], y=yearly_stats['电影数量'],
                                            name='电影数量', line=dict(color='red')), row=2, col=1)
            fig_yearly.add_trace(go.Scatter(x=yearly_stats['上映年份'], y=yearly_stats['平均评分'],
                                            name='平均评分', line=dict(color='purple')), row=2, col=2)

            fig_yearly.update_layout(height=600, showlegend=False, title_text="年度电影市场趋势分析")
            st.plotly_chart(fig_yearly, use_container_width=True)
        else:
            st.warning("年度趋势数据不足以绘制图表。")
    else:
        st.warning("年度趋势分析所需数据（上映年份、票房、评分）不可用。")

    # 类型分析
    st.subheader("电影类型分析")
    if '类型_list' in processed_df.columns and '票房(万元)' in processed_df.columns:
        all_genres = [genre for sublist in processed_df['类型_list'] for genre in sublist]
        genre_counts = pd.Series(all_genres).value_counts().head(15)

        col1, col2 = st.columns(2)
        with col1:
            fig_genre_bar = px.bar(x=genre_counts.values, y=genre_counts.index,
                                   orientation='h', title='电影类型分布',
                                   labels={'x': '电影数量', 'y': '类型'})
            st.plotly_chart(fig_genre_bar, use_container_width=True)

        with col2:
            # 类型平均票房
            genre_box_office = {}
            for idx, genres in processed_df['类型_list'].items():
                box_office = processed_df.loc[idx, '票房(万元)']
                for genre in genres:
                    if genre not in genre_box_office:
                        genre_box_office[genre] = []
                    genre_box_office[genre].append(box_office)

            # 过滤掉出现次数过少的类型，避免异常值影响平均数
            MIN_MOVIES_FOR_AVG = 5
            genre_avg_box = {k: np.mean(v) for k, v in genre_box_office.items() if len(v) >= MIN_MOVIES_FOR_AVG}
            genre_avg_box = dict(sorted(genre_avg_box.items(), key=lambda x: x[1], reverse=True)[:10])

            if genre_avg_box:
                fig_genre_avg = px.bar(x=list(genre_avg_box.values()), y=list(genre_avg_box.keys()),
                                       orientation='h', title=f'各类型平均票房Top10 (至少{MIN_MOVIES_FOR_AVG}部电影)',
                                       labels={'x': '平均票房(万元)', 'y': '类型'})
                st.plotly_chart(fig_genre_avg, use_container_width=True)
            else:
                st.info("没有足够数据计算类型平均票房。")
    else:
        st.warning("类型数据或票房数据不可用。")

    # 月份上映分析
    st.subheader("上映月份分析")
    if '上映月份' in processed_df.columns and '票房(万元)' in processed_df.columns:
        monthly_stats = processed_df.groupby('上映月份').agg(
            平均票房=('票房(万元)', 'mean'),
            电影数量=('电影名', 'count') if '电影名' in processed_df.columns else ('票房(万元)', 'count'),
            平均评分=('豆瓣评分', 'mean')
        ).reset_index()

        col1, col2 = st.columns(2)
        with col1:
            fig_monthly = px.bar(monthly_stats, x='上映月份', y='平均票房',
                                 title='各月份平均票房',
                                 labels={'平均票房': '平均票房(万元)'})
            st.plotly_chart(fig_monthly, use_container_width=True)

        with col2:
            fig_monthly_count = px.bar(monthly_stats, x='上映月份', y='电影数量',
                                       title='各月份上映电影数量')
            st.plotly_chart(fig_monthly_count, use_container_width=True)
    else:
        st.warning("上映月份或票房数据不可用。")

# === 页面3: 票房预测与建模 ===
elif page == "票房预测与建模":
    st.header("🤖 票房预测与建模")

    # 构建模型
    model_results = build_prediction_model(processed_df)

    if model_results:
        # 模型性能对比
        st.subheader("模型性能对比")
        col1, col2 = st.columns(2)

        with col1:
            st.metric("随机森林 R²", f"{model_results['rf_r2']:.3f}")
            st.metric("随机森林 MSE", f"{model_results['rf_mse']:.0f}")

        with col2:
            st.metric("线性回归 R²", f"{model_results['lr_r2']:.3f}")
            st.metric("线性回归 MSE", f"{model_results['lr_mse']:.0f}")

        # 预测结果可视化
        st.subheader("预测结果对比：实际票房 vs. 预测票房")

        # 创建预测vs实际的散点图
        fig_pred = make_subplots(rows=1, cols=2,
                                 subplot_titles=('随机森林预测', '线性回归预测'))

        # 随机森林
        fig_pred.add_trace(go.Scatter(x=model_results['y_test'],
                                      y=model_results['rf_pred'],
                                      mode='markers',
                                      name='RF预测',
                                      opacity=0.6), row=1, col=1)

        # 添加理想预测线
        min_val = min(model_results['y_test'].min(), model_results['rf_pred'].min(), model_results['lr_pred'].min())
        max_val = max(model_results['y_test'].max(), model_results['rf_pred'].max(), model_results['lr_pred'].max())
        fig_pred.add_trace(go.Scatter(x=[min_val, max_val],
                                      y=[min_val, max_val],
                                      mode='lines',
                                      name='理想预测线',
                                      line=dict(dash='dash', color='red')), row=1, col=1)

        # 线性回归
        fig_pred.add_trace(go.Scatter(x=model_results['y_test'],
                                      y=model_results['lr_pred'],
                                      mode='markers',
                                      name='LR预测',
                                      opacity=0.6), row=1, col=2)

        fig_pred.add_trace(go.Scatter(x=[min_val, max_val],
                                      y=[min_val, max_val],
                                      mode='lines',
                                      name='理想预测线',
                                      line=dict(dash='dash', color='red'),
                                      showlegend=False), row=1, col=2)

        fig_pred.update_xaxes(title_text="实际票房(万元)")
        fig_pred.update_yaxes(title_text="预测票房(万元)")
        fig_pred.update_layout(height=500, title_text="预测结果对比", showlegend=True)  # 确保legend显示
        st.plotly_chart(fig_pred, use_container_width=True)

        # 特征重要性（随机森林）
        st.subheader("特征重要性分析 (随机森林)")
        feature_importance = pd.DataFrame({
            'feature': model_results['feature_cols'],
            'importance': model_results['rf_model'].feature_importances_
        }).sort_values('importance', ascending=False).head(20)

        fig_importance = px.bar(feature_importance, x='importance', y='feature',
                                orientation='h', title='Top 20 重要特征',
                                labels={'importance': '重要性', 'feature': '特征'})
        st.plotly_chart(fig_importance, use_container_width=True)

        # 票房预测工具
        st.subheader("🎯 票房预测工具")
        st.write("输入电影参数，预测票房表现：")

        col1, col2, col3 = st.columns(3)

        with col1:
            douban_score = st.slider("豆瓣评分", 1.0, 10.0, 7.0, 0.1)
            duration = st.slider("电影时长(分钟)", 60, 200, 120)
            release_year = st.slider("上映年份", 2010, pd.Timestamp.now().year + 1, pd.Timestamp.now().year)

        with col2:
            douban_comments = st.number_input("豆瓣评论数", 0, 1000000, 10000)
            avg_people = st.slider("场均人数", 1, 50, 15)
            release_month = st.selectbox("上映月份", list(range(1, 13)), index=5)

        with col3:
            is_weekend = st.selectbox("是否周末上映", [0, 1], index=0, format_func=lambda x: "是" if x == 1 else "否")

            # 获取处理后的类型列表，确保用户选择的类型在模型训练时是存在的
            all_processed_types = [col.replace('类型_', '') for col in model_results['feature_cols'] if
                                   col.startswith('类型_')]
            if all_processed_types:
                selected_genre = st.selectbox("主要类型", sorted(list(set(all_processed_types))))
            else:
                st.warning("模型中未找到类型特征，请检查数据预处理。")
                selected_genre = None

        if st.button("预测票房") and selected_genre:
            # 创建预测输入：一个全零的Numpy数组，N为模型训练时特征列的数量
            input_data = {col: 0 for col in model_results['feature_cols']}

            # 设置数值特征
            input_data['豆瓣评分'] = douban_score
            input_data['豆瓣评论数'] = douban_comments
            input_data['时长'] = duration
            input_data['上映年份'] = release_year
            input_data['上映月份'] = release_month
            input_data['是否周末'] = is_weekend
            input_data['场均人数'] = avg_people

            # 设置One-Hot编码的类型特征
            genre_col_name = f'类型_{selected_genre}'
            if genre_col_name in input_data:
                input_data[genre_col_name] = 1

            # 将字典转换为DataFrame的一行
            user_input_df = pd.DataFrame([input_data])

            # 确保列顺序和模型训练时一致
            user_input_df = user_input_df[model_results['feature_cols']]

            # 标准化输入
            pred_input_scaled = model_results['scaler'].transform(user_input_df)

            # 进行预测
            rf_prediction = model_results['rf_model'].predict(pred_input_scaled)[0]
            lr_prediction = model_results['lr_model'].predict(pred_input_scaled)[0]

            # 显示预测结果
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"🌟 随机森林预测票房: {rf_prediction / 10000:.2f} 亿元")
            with col2:
                st.info(f"📊 线性回归预测票房: {lr_prediction / 10000:.2f} 亿元")

            # 预测区间
            avg_prediction = (rf_prediction + lr_prediction) / 2
            st.write(f"📈 综合预测: **{avg_prediction / 10000:.2f} 亿元**")

            # 票房等级判断
            if avg_prediction >= 100000:  # 10亿+
                st.balloons()
                st.success("🎉 预测为超级大片！票房有望超过10亿！")
            elif avg_prediction >= 50000:  # 5亿+
                st.success("🔥 预测为热门电影！票房有望达到5-10亿！")
            elif avg_prediction >= 10000:  # 1亿+
                st.info("👍 预测表现良好，票房有望达到1-5亿")
            else:
                st.warning("📉 预测票房相对较低，可能需要优化营销策略")
        elif not selected_genre:
            st.error("请选择一个主要类型进行预测。")
    else:
        st.warning("模型训练失败，无法进行票房预测。")


# === 页面4: 导演影响力分析 ===
elif page == "导演影响力分析":
    st.header("🎬 导演影响力分析")

    if '导演_list' in processed_df.columns and '票房(万元)' in processed_df.columns and '豆瓣评分' in processed_df.columns:
        all_directors = [d for sublist in processed_df['导演_list'] for d in sublist]
        director_stats = {}

        for director in set(all_directors):
            director_movies = processed_df[processed_df['导演_list'].apply(lambda x: director in x)]
            if len(director_movies) >= 2:  # 至少2部电影，数据更有意义
                director_stats[director] = {
                    '电影数量': len(director_movies),
                    '总票房': director_movies['票房(万元)'].sum(),
                    '平均票房': director_movies['票房(万元)'].mean(),
                    '平均评分': director_movies['豆瓣评分'].mean(),
                    '平均时长': director_movies['时长'].mean(),
                    '总评论数': director_movies['豆瓣评论数'].sum()
                }

        director_df = pd.DataFrame(director_stats).T.reset_index()
        director_df.columns = ['导演', '电影数量', '总票房', '平均票房', '平均评分', '平均时长', '总评论数']
        director_df = director_df.sort_values('总票房', ascending=False)

        if not director_df.empty:
            # 顶级导演排行榜
            st.subheader("📊 导演排行榜")

            tab1, tab2, tab3 = st.tabs(["总票房排行", "平均票房排行", "平均评分排行"])

            with tab1:
                top_directors_total = director_df.head(15)
                fig_director_total = px.bar(top_directors_total, x='总票房', y='导演',
                                            orientation='h', title='导演总票房Top15',
                                            labels={'总票房': '总票房(万元)', '导演': '导演'})
                st.plotly_chart(fig_director_total, use_container_width=True)

            with tab2:
                # 过滤出至少3部电影的导演，使得平均票房更具代表性
                top_directors_avg = director_df[director_df['电影数量'] >= 3].sort_values('平均票房',
                                                                                          ascending=False).head(15)
                fig_director_avg = px.bar(top_directors_avg, x='平均票房', y='导演',
                                          orientation='h', title='导演平均票房Top15 (至少3部电影)',
                                          labels={'平均票房': '平均票房(万元)', '导演': '导演'})
                st.plotly_chart(fig_director_avg, use_container_width=True)

            with tab3:
                top_directors_rating = director_df[director_df['电影数量'] >= 3].sort_values('平均评分',
                                                                                             ascending=False).head(15)
                fig_director_rating = px.bar(top_directors_rating, x='平均评分', y='导演',
                                             orientation='h', title='导演平均评分Top15 (至少3部电影)',
                                             labels={'平均评分': '平均评分', '导演': '导演'})
                st.plotly_chart(fig_director_rating, use_container_width=True)

            # 导演详细分析
            st.subheader("🔍 导演详细分析")

            # 导演选择器
            available_directors = sorted(director_df['导演'].tolist())
            selected_director = st.selectbox("选择导演进行详细分析:", available_directors)

            if selected_director:
                director_movies = processed_df[processed_df['导演_list'].apply(lambda x: selected_director in x)]

                # 导演基本信息
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("电影数量", len(director_movies))
                with col2:
                    st.metric("总票房", f"{director_movies['票房(万元)'].sum() / 10000:.1f}亿")
                with col3:
                    st.metric("平均票房", f"{director_movies['票房(万元)'].mean() / 10000:.2f}亿")
                with col4:
                    st.metric("平均评分", f"{director_movies['豆瓣评分'].mean():.1f}")

                # 导演电影列表
                st.subheader(f"{selected_director} 的电影作品")
                movie_cols = ['电影名', '票房(万元)', '豆瓣评分', '上映日期', '时长']
                available_movie_cols = [col for col in movie_cols if col in director_movies.columns]

                display_movies = director_movies[available_movie_cols].sort_values('票房(万元)', ascending=False)
                st.dataframe(display_movies, use_container_width=True)

                # 导演作品趋势分析
                if len(director_movies) >= 3:  # 至少3部电影才能看趋势
                    st.subheader(f"{selected_director} 作品表现趋势")

                    director_movies_sorted = director_movies.sort_values('上映日期')

                    fig_trend = make_subplots(rows=2, cols=1,
                                              subplot_titles=('票房趋势', '评分趋势'),
                                              vertical_spacing=0.1)

                    fig_trend.add_trace(go.Scatter(x=director_movies_sorted['上映日期'],
                                                   y=director_movies_sorted['票房(万元)'] / 10000,
                                                   mode='lines+markers',
                                                   name='票房(亿)'), row=1, col=1)

                    fig_trend.add_trace(go.Scatter(x=director_movies_sorted['上映日期'],
                                                   y=director_movies_sorted['豆瓣评分'],
                                                   mode='lines+markers',
                                                   name='豆瓣评分',
                                                   line=dict(color='orange')), row=2, col=1)

                    fig_trend.update_layout(height=500, title_text=f"{selected_director} 作品表现趋势",
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
                    '指标': ['电影数量', '总票房(亿)', '平均票房(亿)', '平均评分', '平均时长(分钟)'],
                    director1: [
                        len(director1_movies),
                        director1_movies['票房(万元)'].sum() / 10000,
                        director1_movies['票房(万元)'].mean() / 10000,
                        director1_movies['豆瓣评分'].mean(),
                        director1_movies['时长'].mean()
                    ],
                    director2: [
                        len(director2_movies),
                        director2_movies['票房(万元)'].sum() / 10000,
                        director2_movies['票房(万元)'].mean() / 10000,
                        director2_movies['豆瓣评分'].mean(),
                        director2_movies['时长'].mean()
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
        st.error("导演相关数据（导演列表、票房、评分）不可用，请检查数据预处理。")

# 侧边栏额外信息
st.sidebar.markdown("---")
st.sidebar.markdown("### 📈 数据洞察")

if not processed_df.empty:
    # 显示一些有趣的统计
    # 使用 .get() 安全地访问列，并提供默认值
    highest_grossing_col = '票房(万元)'
    highest_rated_col = '豆瓣评分'
    movie_name_col = '电影名'

    if highest_grossing_col in processed_df.columns and not processed_df[highest_grossing_col].empty:
        highest_grossing = processed_df.loc[processed_df[highest_grossing_col].idxmax()]
        st.sidebar.markdown(f"**最高票房电影:**")
        if movie_name_col in processed_df.columns:
            st.sidebar.write(f"🎬 {highest_grossing.get(movie_name_col, 'N/A')}")
        st.sidebar.write(f"💰 {highest_grossing[highest_grossing_col] / 10000:.1f}亿")
    else:
        st.sidebar.info("最高票房电影数据不可用。")

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
    st.sidebar.write("• 数据来源: 猫眼电影 (假设)")
    st.sidebar.write("• 包含票房、评分、导演、类型等信息")
    st.sidebar.write("• 支持基于历史数据进行票房预测")
else:
    st.sidebar.warning("暂无数据可供分析。")

# 页脚
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>🎬 电影票房分析系统 | 基于机器学习的票房预测 📊</p>
        <p><small>数据驱动决策，助力电影产业发展</small></p>
    </div>
    """,
    unsafe_allow_html=True
)