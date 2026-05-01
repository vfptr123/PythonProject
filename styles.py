"""
全局 CSS 样式常量
=================
将 app.py 中的大段 CSS 抽离，保持 UI 文件专注业务逻辑
"""

# Streamlit 全局注入样式（现代简约 SaaS 风格）
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #fafbfc 0%, #f0f2f5 100%);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

h1 { color: #1a1a2e; font-weight: 700; letter-spacing: -0.5px; }
h2, h3 { color: #2d3748; font-weight: 600; }
h3, h5 { margin-bottom: 0.3rem !important; }

[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"]:has(h3),
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"]:has(h5) {
    margin-bottom: 0 !important;
}

/* 侧边栏 */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: none;
    box-shadow: 2px 0 20px rgba(0,0,0,0.03);
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    border: none; border-radius: 12px; font-weight: 600;
    padding: 12px 20px; transition: all 0.2s ease;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.25);
}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.35);
}
[data-testid="stSidebar"] .stButton > button:disabled {
    background: #e2e8f0; color: #a0aec0; cursor: not-allowed; box-shadow: none;
}
[data-testid="stSidebar"] .stExpander {
    border: none; background: #f8fafc; border-radius: 10px; margin-bottom: 8px;
}

/* 表格 */
.stDataFrame { border: none !important; border-radius: 12px !important; overflow: hidden; box-shadow: 0 2px 12px rgba(0,0,0,0.04) !important; }
.stDataFrame [data-testid="stDataFrameResizable"] { border: none !important; }
.stDataFrame th { background: #f8fafc !important; color: #64748b !important; font-size: 13px !important; font-weight: 600 !important; padding: 14px 16px !important; border: none !important; text-transform: uppercase; letter-spacing: 0.5px; }
.stDataFrame td { font-size: 14px !important; padding: 12px 16px !important; border: none !important; border-bottom: 1px solid #f1f5f9 !important; color: #334155; }
.stDataFrame [data-testid="StyledDataFrameRowContainer"] { border: none !important; }

/* 按钮/上传器 */
.stButton > button { border-radius: 10px; font-weight: 500; transition: all 0.15s ease; }
[data-testid="stFileUploader"] { border: 2px dashed #e2e8f0; border-radius: 12px; padding: 20px; transition: all 0.2s ease; }
[data-testid="stFileUploader"]:hover { border-color: #6366f1; background: #fafaff; }

/* 进度条 */
[data-testid="stProgress"] > div,
[data-testid="stProgressBar"] > div {
    background-color: #f1f5f9 !important; border-radius: 100px !important; height: 5px !important; border: none !important;
}
</style>
"""

# 横向滚动卡片布局（components.html 内嵌样式）
SCROLL_LAYOUT_CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Inter', -apple-system, sans-serif; background: transparent; }
.scroll-container {
    display: flex; gap: 16px; overflow-x: auto;
    padding: 8px 4px 16px 4px; scroll-behavior: smooth;
}
.scroll-container::-webkit-scrollbar { height: 8px; }
.scroll-container::-webkit-scrollbar-track { background: #f1f5f9; border-radius: 10px; }
.scroll-container::-webkit-scrollbar-thumb { background: linear-gradient(90deg, #6366f1, #8b5cf6); border-radius: 10px; }
.image-card {
    flex: 0 0 280px; min-width: 280px; background: #fff;
    border-radius: 16px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.05);
}
.image-card img { width: 100%; height: auto; display: block; }
.card-header { padding: 10px 14px; font-weight: 600; font-size: 13px; letter-spacing: 0.3px; text-align: center; }
.metrics-row { display: flex; gap: 6px; padding: 6px 8px; background: #f8fafc; }
.metric-box { flex: 1; border-radius: 8px; padding: 6px 4px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.03); }
.metric-label { font-size: 10px; font-weight: 500; }
.metric-value { font-size: 15px; font-weight: 700; margin-top: 2px; }
"""
