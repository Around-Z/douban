# modules/data_loader.py (最终完美版)

import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

@st.cache_resource
def get_database_connection():
    """使用 SQLAlchemy 建立并缓存数据库引擎。"""
    try:
        db_uri = 'mysql+pymysql://root:@127.0.0.1/bigwork2025?charset=utf8mb4'
        engine = create_engine(db_uri)
        return engine
    except Exception as e:
        st.error(f"数据库引擎创建失败: {e}");
        return None


@st.cache_data(ttl=3600)
def load_and_prepare_data(_engine):
    if _engine is None: return pd.DataFrame()
    st.info("正在加载票房榜和Top250数据...")
    try:
        df_boxoffice = pd.read_sql("SELECT * FROM maoyan_movies", _engine)
        df_top250 = pd.read_sql("SELECT * FROM douban_top250", _engine)
        if df_boxoffice.empty and df_top250.empty:
            st.warning("两个数据表均为空。");
            return pd.DataFrame()
        st.success(f"成功加载 {len(df_boxoffice)} 条票房榜数据和 {len(df_top250)} 条Top250数据。")
    except Exception as e:
        st.error(f"数据查询失败: {e}");
        return pd.DataFrame()

    # --- 1. 定义一个通用的处理函数，确保逻辑一致 ---
    def process_dataframe(df):
        # 统一列名
        df.rename(columns={'name': 'name_cn', 'douban_score': 'score', 'douban_comment_count': 'comment_count'},
                  inplace=True, errors='ignore')

        # 数值类型转换
        for col in ['year', 'score', 'comment_count', 'box_office_ten_thousand', 'douban_id']:
            if col in df.columns:
                df[col] = pd.to_numeric(df.get(col), errors='coerce')

        # --- 【核心修复】将日期处理逻辑加到这里 ---
        if 'release_date' in df.columns:
            df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
            # 只有当日期转换成功时才提取，否则填充NaN
            df['year'] = df['release_date'].dt.year
            df['month'] = df['release_date'].dt.month
            df['dayofweek'] = df['release_date'].dt.dayofweek  # 0=周一, 6=周日

        # 创建列表列
        for text_col in ['genres_text', 'directors_text', 'actors_text']:
            list_col_name = text_col.replace('_text', '_list')
            if text_col in df.columns:
                df[list_col_name] = df[text_col].fillna('').apply(
                    lambda x: [item.strip() for item in str(x).split(',') if item.strip()]
                )
            else:
                df[list_col_name] = [[] for _ in range(len(df))]
        return df

    # --- 2. 分别处理两个DataFrame ---
    df_boxoffice = process_dataframe(df_boxoffice)
    df_top250 = process_dataframe(df_top250)

    # --- 3. 合并数据 (这部分逻辑不变) ---
    df_top250['is_top250'] = True
    cols_from_top250 = ['douban_id', 'is_top250', 'ranking']
    df_top250_subset = df_top250[[col for col in cols_from_top250 if col in df_top250.columns]]

    df_boxoffice = df_boxoffice.dropna(subset=['douban_id'])
    df_boxoffice['douban_id'] = df_boxoffice['douban_id'].astype(int)
    # top250 的 douban_id 可能有空值，需要先处理
    df_top250 = df_top250.dropna(subset=['douban_id'])
    df_top250['douban_id'] = df_top250['douban_id'].astype(int)
    df_top250_subset = df_top250_subset.dropna(subset=['douban_id'])
    df_top250_subset['douban_id'] = df_top250_subset['douban_id'].astype(int)

    df_merged = pd.merge(df_boxoffice, df_top250_subset, on='douban_id', how='left')
    df_merged['is_top250'].fillna(False, inplace=True)

    # --- 4. 补全Top250中不在票房榜的数据 ---
    top250_not_in_boxoffice = df_top250[~df_top250['douban_id'].isin(df_merged['douban_id'])]

    # 最终合并
    df_final = pd.concat([df_merged, top250_not_in_boxoffice], ignore_index=True)
    # --- 【新增】模型训练模块 ---
    predictor_data = None
    df_model = df_final.dropna(subset=['box_office_ten_thousand', 'score', 'month', 'genres_list']).copy()
    df_model = df_model[df_model['genres_list'].apply(len) > 0]

    if len(df_model) >= 50:  # 确保有足够数据训练
        df_model['main_genre'] = df_model['genres_list'].apply(lambda x: x[0])

        # 特征选择
        numerical_features = ['score']
        categorical_features = ['month', 'main_genre']

        # 确定最常见的类型用于建模，避免过多稀疏特征
        top_genres = df_model['main_genre'].value_counts().nlargest(20).index.tolist()
        df_model_filtered = df_model[df_model['main_genre'].isin(top_genres)]

        # 创建预处理管道
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])

        # 定义模型
        model = RandomForestRegressor(n_estimators=100, random_state=42)

        # 创建完整的Pipeline
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

        # 训练模型
        X = df_model_filtered[numerical_features + categorical_features]
        y = df_model_filtered['box_office_ten_thousand']
        pipeline.fit(X, y)

        # 封装并返回模型和相关信息
        predictor_data = {
            "pipeline": pipeline,
            "genres": top_genres,
            "feature_names": numerical_features + categorical_features
        }

    return df_final, predictor_data