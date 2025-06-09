# app.py
import streamlit as st
import pandas as pd
import  os
# import sys
# from modules.data_loader import get_database_connection, load_and_prepare_data
from modules.page_artistry import render_page_artistry
from modules.data_loader import get_database_connection, load_and_prepare_data
from modules.page_business import render_page_business
from modules.page_directors import render_page_directors
from modules.page_predictor import render_page_predictor
# st.write(f"当前使用的Python解释器: {sys.executable}")
# --- 1. 页面基础配置 ---
st.set_page_config(
    page_title="光影罗盘 (CineCompass)",
    layout="wide",
    page_icon="🧭"
)

# --- 2. 侧边栏与导航 ---
with st.sidebar:
    # --- 顶部Logo与标题 ---
    st.markdown(
        """
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="font-size: 4rem; margin-bottom: 0;">🧭</h1>
            <h2 style="font-family: 'Georgia', 'serif'; ...">光影罗盘</h2>
            <p style="color: #aaa; font-style: italic;">CineCompass</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # --- 导航菜单 ---
    page_options = {
        "🏛️ 艺术殿堂": "artistry",
        "📈 商业版图": "business",
        "🎬 导演影响力中心": "director",
        "🤖 AI票房预测器": "predictor"
    }
    page_labels = list(page_options.keys())

    # --- 【核心修复】使用 Session State ---
    # 1. 初始化session_state，如果它还不存在
    if 'page_id' not in st.session_state:
        st.session_state.page_id = "artistry"  # 默认显示第一个页面

    # 2. 从session_state中获取当前页面的ID，并计算其在列表中的索引
    current_page_id = st.session_state.page_id
    current_page_index = list(page_options.values()).index(current_page_id)

    # 3. 创建radio按钮，并将计算出的索引设为默认值
    selected_page_label = st.radio(
        "**选择探索模块**",
        page_labels,
        index=current_page_index,
        key="main_nav_radio"
    )

    # 4. 当用户点击后，获取新的页面ID，并更新回session_state
    # 这一步是可选的，但可以用来触发一些逻辑。st.radio本身就会触发rerun。
    new_page_id = page_options[selected_page_label]
    if new_page_id != st.session_state.page_id:
        st.session_state.page_id = new_page_id
        # st.rerun() # 通常不需要手动rerun，除非有特殊逻辑

    # --- 【重要】后续代码使用session_state中的值 ---
    selected_page_id = st.session_state.page_id

    st.markdown("---")

    # --- 数据洞察卡片 (代码不变) ---
    st.markdown("""<div style="font-size: 1.2em; color: black; font-weight: bold;">Python 爬虫 可视化 期末大作业</div>""", unsafe_allow_html=True)
    # ... (st.metric 代码)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")

    # --- 页脚 (代码不变) ---
    # st.markdown("""<div style="font-size: 1.5em; color: black;">Python 爬虫 可视化 期末大作业</div>""", unsafe_allow_html=True)
    # st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #888; font-size: 1.5em;">
            <p>23211410224 周为洲</p>
            <!-- 这是一个标准的HTML超链接标签 -->
            <a 
                href="https://github.com/Around-Z/douban/" 
                target="_blank" 
                style="color: #FFD700; text-decoration: none;"
            >
                项目代码 GitHub 🚀
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )
# --- 3. 数据加载 ---
# 只在需要时加载数据
# db_engine = get_database_connection() # 变量名改为 db_engine 更清晰
# df_top250 = pd.DataFrame() # 初始化为空
db_conn = get_database_connection()
df_processed = pd.DataFrame() # 使用一个通用的变量名
if db_conn:
    # 无论哪个页面，我们都加载这个大而全的df_full
    df_processed, predictor_data = load_and_prepare_data(db_conn)
    if df_processed.empty:
        st.error("数据加载失败或数据为空，无法渲染页面。")
        st.stop()
else:
    st.error("无法连接到数据库，请检查服务是否启动。")
    st.stop()
# --- 4. 页面路由 ---
if selected_page_id == "artistry":
    df_top250 = df_processed[df_processed['is_top250'] == True].copy()
    render_page_artistry(df_top250)
elif selected_page_id == "business":
    # st.title("📈 商业版图 (票房分析)")
    # st.info("此模块正在建设中...")
    render_page_business(df_processed)
elif selected_page_id == "director":
    # st.title("🎬 导演影响力中心")
    # st.info("此模块正在建设中...")
    render_page_directors(df_processed)
elif selected_page_id == "predictor":
    render_page_predictor(df_processed, predictor_data)