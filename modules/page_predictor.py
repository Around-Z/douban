# modules/page_predictor.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def render_page_predictor(df_full, predictor_data):
    """渲染“AI票房预测器”页面的主函数。"""
    st.title("🔭 电影票房潜力探测器")
    st.caption("选择影片的核心基因，看看它在市场中能激起多大浪花！")

    if not predictor_data:
        st.warning("模型训练数据不足或失败，预测器无法启动。");
        return

    pipeline = predictor_data["pipeline"]
    genre_list = sorted(predictor_data["genres"])

    # --- 三步式输入面板 ---
    st.header("预测参数输入")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("① 口碑定调")
        douban_score = st.slider("预估豆瓣评分", 5.0, 9.5, 7.0, 0.1)
    with col2:
        st.subheader("② 选择类型")
        selected_genre = st.selectbox("核心电影类型", genre_list)
    with col3:
        st.subheader("③ 瞄准档期")
        selected_month = st.select_slider("选择上映月份", options=list(range(1, 13)), value=7,
                                          format_func=lambda x: f"{x}月")

    # --- 预测启动按钮 ---
    if st.button("🧬 解码票房潜力！", use_container_width=True, type="primary"):
        with st.spinner("AI正在进行深度计算..."):
            # 准备基准输入
            baseline_input = pd.DataFrame([{
                'score': 7.0,
                'month': 6,
                'main_genre': '剧情'
            }])
            baseline_prediction = pipeline.predict(baseline_input)[0]

            # 准备用户输入
            user_input = pd.DataFrame([{
                'score': douban_score,
                'month': selected_month,
                'main_genre': selected_genre
            }])
            total_prediction = pipeline.predict(user_input)[0]

            # --- 贡献度简化计算 ---
            # 口碑贡献
            score_input = baseline_input.copy();
            score_input['score'] = douban_score
            score_pred = pipeline.predict(score_input)[0]
            score_contribution = score_pred - baseline_prediction

            # 类型贡献
            genre_input = baseline_input.copy();
            genre_input['main_genre'] = selected_genre
            genre_pred = pipeline.predict(genre_input)[0]
            genre_contribution = genre_pred - baseline_prediction

            # 档期贡献
            month_input = baseline_input.copy();
            month_input['month'] = selected_month
            month_pred = pipeline.predict(month_input)[0]
            month_contribution = month_pred - baseline_prediction

            # --- 可视化结果 ---
            st.success(f"综合潜力预测: **{total_prediction / 10000:.2f} 亿**")

            st.subheader("票房潜力构成图")

            contributions = {
                '口碑加成': score_contribution,
                '类型加成': genre_contribution,
                '档期加成': month_contribution
            }

            fig = go.Figure(go.Bar(
                y=['票房构成'],
                x=[baseline_prediction],
                name='基准票房',
                orientation='h',
                marker=dict(color='grey')
            ))

            for name, value in contributions.items():
                fig.add_trace(go.Bar(
                    y=['票房构成'],
                    x=[value],
                    name=name,
                    orientation='h',
                    marker=dict(color='green' if value >= 0 else 'red')
                ))

            fig.update_layout(barmode='relative', title_text="预测票房构成分析 (万元)", height=250)
            st.plotly_chart(fig, use_container_width=True)

            # --- AI洞察解读 ---
            with st.expander("查看AI的决策分析"):
                st.markdown("🔍 **AI洞察:**")
                st.markdown(
                    f"- **口碑力量**: 你的 **{douban_score}分** 设定，为票房潜力带来了约 **{score_contribution / 10000:.2f}亿** 的调整！")
                st.markdown(
                    f"- **类型赛道**: 选择 **'{selected_genre}'** 类型，相比基准类型，带来了约 **{genre_contribution / 10000:.2f}亿** 的调整。")
                st.markdown(
                    f"- **档期智慧**: 在 **{selected_month}月** 上映，档期因素影响了约 **{month_contribution / 10000:.2f}亿**。")