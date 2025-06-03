

# --- Streamlit 页面配置 (必须是第一个 Streamlit 命令) ---
st.set_page_config(layout="wide", page_title="电影票房分析与预测")

# 设置 Matplotlib 字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或者 'Microsoft YaHei'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# --- 数据库连接 (使用 Streamlit 缓存资源，避免每次运行都重新连接) ---
@st.cache_resource
def get_database_connection_resource():
    try:
        db = pymysql.connect(host='localhost',
                             user='root',
                             password='',
                             database='douban',
                             charset='utf8mb4')
        return db
    except pymysql.err.OperationalError as e:
        st.exception(f"数据库连接失败 (请检查MySQL服务): {e}")
        st.stop()
    except Exception as e:
        st.exception(f"发生未知数据库错误: {e}")
        st.stop()


db_connection = get_database_connection_resource()


# --- 数据加载与预处理 (使用 Streamlit 缓存数据) ---
@st.cache_data
def load_and_preprocess_data(_db_conn) -> pd.DataFrame:
    st.info("正在加载和预处理电影数据...")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            df = pd.read_sql(f'SELECT * FROM `maoyan_movie`', _db_conn)
        st.success(f"成功加载 {len(df)} 行数据从 'maoyan_movie'.")
        if df.empty:
            st.warning("maoyan_movie 表中没有数据，请先运行爬虫填充数据。")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"从表 'maoyan_movie' 加载数据失败: {e}")
        return pd.DataFrame()

    df = df.copy()

    # 1. 上映日期处理
    df['上映日期'] = pd.to_datetime(df['上映日期'], errors='coerce')
    if df['上映日期'].isnull().any():
        st.warning("警告: '上映日期' 列中存在无法解析的日期，已转换为 NaT。")

    df['上映年份'] = df['上映日期'].dt.year.fillna(-1).astype(int)
    df['上映月份'] = df['上映日期'].dt.month.fillna(-1).astype(int)
    df['上映日'] = df['上映日期'].dt.day.fillna(-1).astype(int)
    df['上映星期几'] = df['上映日期'].dt.dayofweek.fillna(-1).astype(int)  # 0=星期一, 6=星期日
    df['是否周末'] = df['上映星期几'].isin([5, 6]).astype(int)

    # 2. 处理多值分类特征 ('导演', '演员', '类型', '地区', '语言')
    original_multi_value_cols = ['导演', '演员', '类型', '地区', '语言']
    for col in original_multi_value_cols:
        # 创建 '_list' 结尾的新列来存储列表形式的数据，这些列会保留用于EDA
        df[col + '_list'] = df[col].apply(
            lambda x: [item.strip() for item in str(x).split(',') if item.strip()] if pd.notnull(x) else []
        )

    # 3. 转换 '豆瓣评分' 和 '时长' 为数值
    df['豆瓣评分'] = pd.to_numeric(df['豆瓣评分'], errors='coerce').fillna(0.0)
    df['时长'] = pd.to_numeric(df['时长'], errors='coerce').fillna(0).astype(int)

    # 过滤掉不适合分析或预测的电影
    current_year = pd.Timestamp.now().year
    initial_rows = len(df)
    df = df[df['上映年份'] <= current_year]
    df = df[df['票房(万元)'] > 0]
    df = df[df['豆瓣评分'] > 0]
    df = df[df['豆瓣评论数'] > 0]
    df = df[df['时长'] > 0]

    st.write(
        f"已过滤掉 {initial_rows - len(df)} 行不适合分析或预测的电影（未来上映/票房0/评分0/评论数0/时长0）。剩余 {len(df)} 行。")

    # --- 4. 对选定特征进行 One-Hot 编码 ---
    # 演员列通常维度过高，不进行 One-Hot 编码。
    ohe_target_cols = ['导演', '类型', '地区', '语言']

    all_directors = [d for sublist in df['导演_list'] for d in sublist]
    all_types = [t for sublist in df['类型_list'] for t in sublist]
    all_places = [p for sublist in df['地区_list'] for p in sublist]
    all_langs = [l for sublist in df['语言_list'] for l in sublist]

    TOP_N_DIRECTORS = 30
    TOP_N_TYPES = 15
    TOP_N_PLACES = 10
    TOP_N_LANGS = 10

    top_directors = pd.Series(all_directors).value_counts().head(TOP_N_DIRECTORS).index.tolist()
    top_types = pd.Series(all_types).value_counts().head(TOP_N_TYPES).index.tolist()
    top_places = pd.Series(all_places).value_counts().head(TOP_N_PLACES).index.tolist()
    top_langs = pd.Series(all_langs).value_counts().head(TOP_N_LANGS).index.tolist()

    def one_hot_encode_multi_value(df_col_list, top_items_list, prefix):
        temp_df = pd.DataFrame(0, index=df_col_list.index, columns=[f"{prefix}_{item}" for item in top_items_list])
        for idx, items in df_col_list.items():
            for item in items:
                if item in top_items_list:
                    temp_df.loc[idx, f"{prefix}_{item}"] = 1
        return temp_df

    st.write("正在进行 One-Hot 编码...")
    df_directors_ohe = one_hot_encode_multi_value(df['导演_list'], top_directors, '导演')
    df_types_ohe = one_hot_encode_multi_value(df['类型_list'], top_types, '类型')
    df_places_ohe = one_hot_encode_multi_value(df['地区_list'], top_places, '地区')
    df_langs_ohe = one_hot_encode_multi_value(df['语言_list'], top_langs, '语言')

    # 将编码后的 DataFrame 合并回主 DataFrame
    df = pd.concat([df, df_directors_ohe, df_types_ohe, df_places_ohe, df_langs_ohe], axis=1)

    # --- 5. 清理原始字符串列 ---
    # 删除原始的字符串列，因为它们的值已经被提取到 _list 列或 OHE 列中
    # 这样，在 DataFrame 中就不会有原始的字符串列和它们的 '_list' 形式同时存在
    cols_to_drop_original_string = [col for col in original_multi_value_cols if col in df.columns]
    df.drop(columns=cols_to_drop_original_string, inplace=True)

    st.success("数据预处理和特征工程完成。")
    return df


# --- 导演影响力雷达图辅助函数 ---
def plot_director_radar_chart(director_name, processed_df):
    director_movies = processed_df[processed_df['导演_list'].apply(lambda x: director_name in x)]

    if director_movies.empty:
        st.warning(f"数据集中未找到导演 '{director_name}' 的电影信息。")
        return

    total_movies = len(director_movies)
    avg_box_office = director_movies['票房(万元)'].mean()
    avg_douban_score = director_movies['豆瓣评分'].mean()
    avg_length = director_movies['时长'].mean()
    avg_avg_people = director_movies['场均人数'].mean()

    overall_avg_box_office = processed_df['票房(万元)'].mean()
    overall_avg_douban_score = processed_df['豆瓣评分'].mean()
    overall_avg_length = processed_df['时长'].mean()
    overall_avg_avg_people = processed_df['场均人数'].mean()

    all_directors_flat = [d for sublist in processed_df['导演_list'] for d in sublist]
    unique_directors_count = len(pd.Series(all_directors_flat).unique())
    overall_total_movies_per_director = len(processed_df) / unique_directors_count if unique_directors_count > 0 else 1

    categories = ['电影数量', '平均票房', '平均评分', '平均片长', '场均人数']
    num_vars = len(categories)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    epsilon = 1e-6
    values_director = [
        total_movies / (overall_total_movies_per_director + epsilon),
        avg_box_office / (overall_avg_box_office + epsilon),
        avg_douban_score / (overall_avg_douban_score + epsilon),
        avg_length / (overall_avg_length + epsilon),
        avg_avg_people / (overall_avg_avg_people + epsilon)
    ]
    values_director = [min(v, 2.0) for v in values_director]
    values_director += values_director[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, values_director, color='red', alpha=0.25)
    ax.plot(angles, values_director, color='red', linewidth=2, linestyle='solid', label=director_name)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color='grey', size=12)

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

        if value * 1.15 > ax.get_rmax():
            ax.set_rmax(value * 1.2)

        ax.text(angle, value * 1.15, text_val, color='black', size=10,
                horizontalalignment='center' if angle % (pi / 2) == 0 else ('left' if 0 < angle < pi else 'right'),
                verticalalignment='center' if angle == 0 else ('bottom' if 0 < angle < pi else 'top'))

    ax.set_title(f'导演 {director_name} 影响力雷达图', va='bottom', fontsize=16)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    st.pyplot(fig)


# --- 主应用程序逻辑 ---
st.title("电影票房预测与观众偏好分析 🎬")
st.markdown("---")

st.sidebar.header("导航")
page = st.sidebar.radio("选择页面", ["数据概览", "探索性分析", "票房预测与建模", "导演影响力分析"])

processed_df = load_and_preprocess_data(db_connection)
