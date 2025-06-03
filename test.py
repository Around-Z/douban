from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

# --- 请在这里配置您的字体路径 ---
# 示例 1: Windows 默认黑体
# font_path = r'C:\Windows\Fonts\simhei.ttf'

# 示例 2: Windows 默认微软雅黑
font_path = r'C:\Windows\Fonts\FZSTK.TTF'

# 示例 3: 如果 SimHei.ttf 放在和这个脚本相同的目录下
# current_dir = os.path.dirname(os.path.abspath(__file__))
# font_path = os.path.join(current_dir, 'SimHei.ttf')
# --- 确保你只启用其中一行，并且路径是正确的 ---


# 检查字体文件是否存在
if not os.path.exists(font_path):
    print(f"错误: 字体文件不存在于路径: {font_path}")
    print("请检查字体路径是否正确，或将字体文件放到正确的位置。")
    exit()

text = "你好 世界 Python 编程 词云 可视化 数据分析 机器 学习 电影 票房 豆瓣 评价 导演 类型 地区 语言 上映 日期 时长 语录 主演 艺术 生活 梦想 真实 故事 精彩 瞬间"

try:
    wc = WordCloud(
        font_path=font_path,
        background_color="white",
        max_words=200,
        width=1000,
        height=600,
        margin=2,
        random_state=42,
        collocations=False # 不包含词组
    )
    wc.generate(text)

    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title("测试词云", fontsize=18)
    plt.show() # 显示图片
    print("词云生成成功，请查看弹出的图像窗口。")

except Exception as e:
    print(f"生成词云时发生错误: {e}")
    print("请根据错误信息进一步排查。")