# modules/visualizations.py
import streamlit as st
import jieba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np # 导入numpy
from PIL import Image
from urllib.parse import quote_plus
FONT_PATH = 'assets/simhei.ttf'  # 确保字体文件与主脚本在同一目录


# --- 【新增】图片URL代理函数 ---
@st.cache_data
def get_proxied_image_url(url):
    """
    使用 images.weserv.nl 代理图片URL，以解决防盗链和HTTP问题。
    """
    # 检查URL是否有效
    if pd.isna(url) or not isinstance(url, str) or not url.startswith('http'):
        # 如果URL无效，返回一个占位图
        return "https://via.placeholder.com/200x300.png?text=Image+Not+Found"

    # 将原始URL进行URL编码，确保特殊字符能被正确传递
    encoded_url = quote_plus(url)

    # 构建代理URL
    return f"https://images.weserv.nl/?url={encoded_url}"

@st.cache_data
def generate_wordcloud_text(text_series, stopwords_path="assets/stopwords.txt"):
    """对Pandas Series中的文本进行分词和停用词过滤，返回一个用于生成词云的字符串。"""
    try:
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            stopwords = set([line.strip() for line in f])
    except FileNotFoundError:
        st.warning(f"停用词文件未找到: {stopwords_path}。将使用内置的简单停用词。")
        stopwords = {'的', '是', '了', '和', '也', '在', '我', '你', '他'}

    # 合并所有文本并分词
    full_text = ' '.join(text_series.dropna())
    words = jieba.lcut(full_text)

    # 过滤停用词和短词
    filtered_words = [word for word in words if len(word) > 1 and word not in stopwords]

    return " ".join(filtered_words)


# modules/visualizations.py

# ... 其他 imports ...

def create_and_show_wordcloud(processed_text, colormap='YlOrBr'):
    if not processed_text: st.info("没有足够的文本来生成词云。"); return
    font_path = 'assets/STXINGKA.ttf'
    if not os.path.exists(font_path): st.error(f"字体文件 '{font_path}' 未找到！"); return

    wordcloud = WordCloud(
        font_path=font_path, background_color=None, mode="RGBA",
        width=1600, height=500, scale=2, colormap=colormap,
        max_words=150, min_font_size=12, max_font_size=120,
        relative_scaling=0.4, prefer_horizontal=0.98,
        random_state=42, collocations=False,
    ).generate(processed_text)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(wordcloud, interpolation='antialiased');
    ax.axis('off');
    plt.tight_layout(pad=0)
    st.pyplot(fig, use_container_width=True, clear_figure=True)