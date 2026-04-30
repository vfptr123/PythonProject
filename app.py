"""
低照度图像增强评估系统 v5.0
===============================
核心重构：
1. 解耦业务逻辑：指标计算、算法调度移至 image_processing.py
2. app.py 仅负责 UI 展示和接口调用
3. 模块职责清晰，便于维护和扩展
"""
import streamlit as st
import cv2
import numpy as np
import base64
import time
import image_processing

# ==========================================
# 1. 页面配置（必须是第一个 Streamlit 命令）
# ==========================================
st.set_page_config(
    page_title="低照度图像增强评估系统",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. 全局样式 - 现代简约 SaaS 风格
# ==========================================
st.markdown("""
<style>
/* ========== 基础背景与字体 ========== */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

[data-testid="stAppViewContainer"] {
    background: linear-gradient(180deg, #fafbfc 0%, #f0f2f5 100%);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* 标题样式 */
h1 { 
    color: #1a1a2e; 
    font-weight: 700; 
    letter-spacing: -0.5px;
}
h2, h3 { 
    color: #2d3748; 
    font-weight: 600; 
}

/* 紧凑标题间距 */
h3, h5 {
    margin-bottom: 0.3rem !important;
}

/* 主区域标题与内容紧凑 */
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"]:has(h3),
[data-testid="stVerticalBlock"] > [data-testid="stVerticalBlockBorderWrapper"]:has(h5) {
    margin-bottom: 0 !important;
}

/* ========== 侧边栏极简风 ========== */
[data-testid="stSidebar"] { 
    background: #ffffff; 
    border-right: none;
    box-shadow: 2px 0 20px rgba(0,0,0,0.03);
}

[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    border: none; 
    border-radius: 12px; 
    font-weight: 600;
    padding: 12px 20px;
    transition: all 0.2s ease;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.25);
}

[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.35);
}

[data-testid="stSidebar"] .stButton > button:disabled {
    background: #e2e8f0; 
    color: #a0aec0; 
    cursor: not-allowed;
    box-shadow: none;
}

/* 侧边栏折叠面板 */
[data-testid="stSidebar"] .stExpander {
    border: none;
    background: #f8fafc;
    border-radius: 10px;
    margin-bottom: 8px;
}

/* ========== 表格无界化样式 ========== */
.stDataFrame {
    border: none !important;
    border-radius: 12px !important;
    overflow: hidden;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04) !important;
}

.stDataFrame [data-testid="stDataFrameResizable"] {
    border: none !important;
}

/* 表头样式 */
.stDataFrame th {
    background: #f8fafc !important;
    color: #64748b !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 14px 16px !important;
    border: none !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* 表格单元格 */
.stDataFrame td {
    font-size: 14px !important;
    padding: 12px 16px !important;
    border: none !important;
    border-bottom: 1px solid #f1f5f9 !important;
    color: #334155;
}

/* 隐藏表格网格线 */
.stDataFrame [data-testid="StyledDataFrameRowContainer"] {
    border: none !important;
}

/* ========== 按钮统一风格 ========== */
.stButton > button {
    border-radius: 10px;
    font-weight: 500;
    transition: all 0.15s ease;
}

/* ========== 上传器美化 ========== */
[data-testid="stFileUploader"] {
    border: 2px dashed #e2e8f0;
    border-radius: 12px;
    padding: 20px;
    transition: all 0.2s ease;
}

[data-testid="stFileUploader"]:hover {
    border-color: #6366f1;
    background: #fafaff;
}

/* ========== 进度条“极简灰 -> 翡翠绿”终极版 ========== */

/* 1. 底层轨道：极淡的浅灰色 */
[data-testid="stProgress"] > div,
[data-testid="stProgressBar"] > div {
    background-color: #f1f5f9 !important; /* 极淡的灰底色 */
    border-radius: 100px !important;
    height: 5px !important; /* 压细一点更精致 */
    border: none !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. 状态初始化
# ==========================================
def init_session_state():
    defaults = {
        # UI 状态：'upload' / 'result' / 'changing'（更换图片中）
        "ui_state": "upload",

        # 算法选择
        "selected_algorithms": [],

        # 处理标志
        "processing": False,

        # 结果缓存
        "results": {},
        "metrics": {},
        "performance": {},

        # 图像缓存 - 低照度输入图
        "img_bgr": None,
        "img_rgb": None,
        "uploaded_filename": "",
        "file_bytes": None,
        
        # 参考图像缓存 (Ground Truth) - 可选
        "ref_bgr": None,
        "ref_rgb": None,
        "ref_filename": "",
        "ref_file_bytes": None,

        # 算法参数
        "he_params": {"clip_limit": 3.0, "tile_grid": 8},
        "retinex_params": {
            "sigma_list": "15,80,250",
            "G": 5.0, "b": 25.0,
            "alpha": 125.0, "beta": 46.0,
            "low_clip": 0.01, "high_clip": 0.99
        },
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ==========================================
# 4. 缓存包装器（调用 image_processing 接口）
# ==========================================
@st.cache_data(max_entries=20, show_spinner=False)
def calculate_metrics_cached(img1_bytes: bytes, img2_bytes: bytes) -> dict:
    """带缓存的指标计算（调用 image_processing 接口）"""
    return image_processing.calculate_metrics_from_bytes(img1_bytes, img2_bytes)

# ==========================================
# 5. 前置处理文件上传状态
# ==========================================

# 创建主区域顶部占位符（用于进度条）
main_progress_placeholder = st.empty()

# 根据当前 UI 状态，前置处理文件上传
current_uploaded_file = None

if st.session_state.ui_state == "upload":
    # 状态 A：显示标题和上传器（紧凑布局）
    st.markdown("## 低照度图像增强评估系统")

    # === 必选：低照度待处理图像 ===
    st.markdown("##### 低照度图像（必选）")
    st.caption("上传图片并选择算法对比增强效果")
    current_uploaded_file = st.file_uploader(
        "选择低照度图像",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        label_visibility="collapsed",
        key="uploader_state_a"
    )

    # 立即处理上传的文件，更新 session_state
    if current_uploaded_file is not None:
        file_bytes = current_uploaded_file.getvalue()
        img_bgr = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

        if img_bgr is not None:
            st.session_state.img_bgr = img_bgr
            st.session_state.img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            st.session_state.uploaded_filename = current_uploaded_file.name
            st.session_state.file_bytes = file_bytes

    # === 可选：正常曝光参考图 (Ground Truth) ===
    st.markdown("##### 参考图像（可选）")
    st.caption("用于计算 RMSE/PSNR/SSIM")
    ref_uploaded_file = st.file_uploader(
        "选择参考图像",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        label_visibility="collapsed",
        key="uploader_ref_a"
    )

    # 处理参考图上传
    if ref_uploaded_file is not None:
        ref_bytes = ref_uploaded_file.getvalue()
        ref_bgr = cv2.imdecode(np.frombuffer(ref_bytes, np.uint8), cv2.IMREAD_COLOR)
        if ref_bgr is not None:
            st.session_state.ref_bgr = ref_bgr
            st.session_state.ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
            st.session_state.ref_filename = ref_uploaded_file.name
            st.session_state.ref_file_bytes = ref_bytes
    else:
        # 用户取消或未上传参考图时清空
        if st.session_state.ref_bgr is not None and ref_uploaded_file is None:
            # 保持原状，除非用户主动清除
            pass

elif st.session_state.ui_state == "changing":
    # 状态 C（更换图片中）：在结果上方显示新的上传器
    st.markdown("## 更换图片")
    st.info("请选择新的图片文件。上传成功后，旧的对比结果将被清空。")

    # === 必选：低照度待处理图像 ===
    st.markdown("##### 低照度图像（必选）")
    new_uploaded_file = st.file_uploader(
        "选择新图像",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        label_visibility="collapsed",
        key="uploader_change"
    )

    # === 可选：正常曝光参考图 ===
    st.markdown("##### 参考图像（可选）")
    new_ref_file = st.file_uploader(
        "选择参考图像",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        label_visibility="collapsed",
        key="uploader_ref_change"
    )

    col_cancel, col_spacer = st.columns([1, 3])
    with col_cancel:
        if st.button("❌ 取消更换", use_container_width=True):
            st.session_state.ui_state = "result"
            st.rerun()

    # 如果用户上传了新图片
    if new_uploaded_file is not None:
        file_bytes = new_uploaded_file.getvalue()
        img_bgr = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)

        if img_bgr is not None:
            # 清空旧结果，更新为新图片
            st.session_state.results = {}
            st.session_state.metrics = {}
            st.session_state.performance = {}
            st.session_state.img_bgr = img_bgr
            st.session_state.img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            st.session_state.uploaded_filename = new_uploaded_file.name
            st.session_state.file_bytes = file_bytes
            
            # 处理参考图（可选）
            if new_ref_file is not None:
                ref_bytes = new_ref_file.getvalue()
                ref_bgr = cv2.imdecode(np.frombuffer(ref_bytes, np.uint8), cv2.IMREAD_COLOR)
                if ref_bgr is not None:
                    st.session_state.ref_bgr = ref_bgr
                    st.session_state.ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
                    st.session_state.ref_filename = new_ref_file.name
                    st.session_state.ref_file_bytes = ref_bytes
            else:
                # 用户未上传参考图，清空旧的参考图
                st.session_state.ref_bgr = None
                st.session_state.ref_rgb = None
                st.session_state.ref_filename = ""
                st.session_state.ref_file_bytes = None
            
            st.session_state.ui_state = "upload"  # 回到上传状态（准备就绪）
            st.rerun()

    st.markdown("---")

# ==========================================
# 7. 侧边栏（在文件上传处理之后渲染）
# ==========================================
with st.sidebar:
    st.markdown("## 算法控制台")
    st.markdown("<hr style='margin:8px 0;border:none;border-top:1px solid #e2e8f0;'>", unsafe_allow_html=True)

    st.markdown("### 选择算法（可多选）")

    algo_info = {
        "HE": ("HE", "全局直方图均衡化"),
        "Retinex": ("Retinex", "多尺度 Retinex 增强"),
        "Zero-DCE": ("Zero-DCE", "零参考深度学习")
    }

    selected = []
    for algo, (emoji_name, desc) in algo_info.items():
        if st.checkbox(f"{emoji_name} - {desc}", value=algo in st.session_state.selected_algorithms, key=f"chk_{algo}"):
            selected.append(algo)
    st.session_state.selected_algorithms = selected

    # 参数配置 - 紧凑并排布局 + 气泡提示
    if "HE" in selected:
        with st.expander("HE 参数", expanded=False):
            st.caption("HE 为全局均衡，无需额外参数")

    if "Retinex" in selected:
        with st.expander("Retinex 参数", expanded=False):
            p = st.session_state.retinex_params
            # Sigma List - 显示标签 + 问号提示
            p["sigma_list"] = st.text_input(
                "Sigma List", p["sigma_list"],
                placeholder="例: 15,80,250",
                help="高斯模糊尺度列表（英文逗号隔开）\n\n小尺度：保留细节\n\n中尺度：平衡细节与光照\n\n大尺度：估计全局光照\n\n📌 参考值：15,80,250"
            )
            
            # 增益 G 和 偏差 b
            col_g, col_b = st.columns(2)
            with col_g:
                p["G"] = st.number_input(
                    "G", 1.0, 20.0, p["G"], 0.5, format="%.1f",
                    help="增益系数\n⬆️ 调大：图像整体变亮\n📌 参考值：5.0"
                )
            with col_b:
                p["b"] = st.number_input(
                    "b", 1.0, 100.0, p["b"], 1.0, format="%.1f",
                    help="偏移量\n⬆️ 调大：暗部提亮更明显\n📌 参考值：25.0"
                )
            
            # Alpha 和 Beta
            col_a, col_bt = st.columns(2)
            with col_a:
                p["alpha"] = st.number_input(
                    "α", 50.0, 300.0, p["alpha"], 5.0, format="%.0f",
                    help="颜色恢复强度\n⬆️ 调大：色彩更鲜艳饱和\n📌 参考值：125"
                )
            with col_bt:
                p["beta"] = st.number_input(
                    "β", 10.0, 100.0, p["beta"], 1.0, format="%.0f",
                    help="颜色恢复偏移\n配合 α 微调色彩平衡\n📌 参考值：46"
                )
            
            # 截断参数 - 改用数值输入
            col_lo, col_hi = st.columns(2)
            with col_lo:
                p["low_clip"] = st.number_input(
                    "暗部截断", 0.0, 0.1, p["low_clip"], 0.01, format="%.2f",
                    help="暗部截断比例\n⬆️ 调大：去除更多暗部噪点\n📌 参考值：0.01"
                )
            with col_hi:
                p["high_clip"] = st.number_input(
                    "亮部截断", 0.9, 1.0, p["high_clip"], 0.01, format="%.2f",
                    help="亮部截断比例\n⬇️ 调小：抑制过曝区域\n📌 参考值：0.99"
                )

    # 深度学习模型说明 - 简洁版
    if "Zero-DCE" in selected:
        with st.expander("深度学习说明", expanded=False):
            st.caption("首次加载需要 5-15 秒，之后会缓存")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    # 【修复】现在 session_state 已经是最新的，按钮状态即时正确
    has_image = st.session_state.img_bgr is not None
    has_algo = len(selected) > 0
    is_processing = st.session_state.processing

    can_execute = has_image and has_algo and not is_processing

    if not has_image:
        st.caption("⚠️ 请先上传图片")
    elif not has_algo:
        st.caption("⚠️ 请至少选择一个算法")

    execute_btn = st.button(
        "🚀 执行选中算法",
        type="primary",
        use_container_width=True,
        disabled=not can_execute
    )

# ==========================================
# 8. 主区域内容渲染
# ==========================================

# -------- 状态 A: 上传准备阶段（继续渲染） --------
if st.session_state.ui_state == "upload":
    if st.session_state.img_bgr is not None:
        # 【优化】紧凑单行布局，减少间距
        h, w = st.session_state.img_bgr.shape[:2]
        file_size_kb = len(st.session_state.file_bytes) / 1024 if st.session_state.file_bytes else 0
        
        st.success(f"✅ **低照度图片已就绪**  |  `{st.session_state.uploaded_filename}`  |  {w}×{h} px  |  {file_size_kb:.1f} KB")
        
        # 参考图信息（如果有）
        if st.session_state.ref_bgr is not None:
            ref_h, ref_w = st.session_state.ref_bgr.shape[:2]
            ref_size_kb = len(st.session_state.ref_file_bytes) / 1024 if st.session_state.ref_file_bytes else 0
            st.info(f"✅ **参考图已就绪**  |  `{st.session_state.ref_filename}`  |  {ref_w}×{ref_h} px  |  {ref_size_kb:.1f} KB")
        else:
            st.caption("⚠️ 未上传参考图，RMSE/PSNR/SSIM 指标将显示为 N/A")
        
        st.markdown("<p style='margin:4px 0 0 0;font-size:13px;color:#6366f1;'>👈 请在左侧侧边栏选择算法，然后点击「执行选中算法」按钮</p>", unsafe_allow_html=True)
    else:
        st.markdown("---")
        st.markdown("#### 📝 使用说明")
        st.markdown("""
        1. **上传图像**：支持 PNG、JPG、JPEG、BMP 格式
        2. **选择算法**：在左侧勾选需要对比的算法（可多选）
        3. **配置参数**：根据需要调整算法参数
        4. **执行对比**：点击「执行选中算法」进行并行处理
        5. **查看结果**：对比各算法的处理效果和客观指标
        """)

# -------- 状态 B / C: 结果展示阶段 --------
elif st.session_state.ui_state in ["result", "changing"]:
    # 如果是 changing 状态，上面已经渲染了上传器，这里继续显示旧结果
    if st.session_state.ui_state == "result":
        st.markdown("<p style='font-size:20px;font-weight:600;color:#2d3748;margin:0 0 8px 0;'>多算法对比结果</p>", unsafe_allow_html=True)
    else:
        st.markdown("<p style='font-size:20px;font-weight:600;color:#2d3748;margin:0 0 8px 0;'>当前对比结果（即将被替换）</p>", unsafe_allow_html=True)

    # 算法信息映射
    algo_badge_map = {
        "HE": ("#ebf8ff", "#2b6cb0"),
        "Retinex": ("#faf5ff", "#553c9a"),
        "Zero-DCE": ("#fff5f5", "#c53030")
    }
    algo_name_map = {
        "原图": "原图",
        "HE": "HE 直方图均衡",
        "Retinex": "Retinex 增强",
        "Zero-DCE": "Zero-DCE 深度学习"
    }

    # 获取所有算法（包括原图）- 按侧边栏顺序排列
    ALGO_ORDER = ["HE", "Retinex", "Zero-DCE"]
    sorted_algos = [a for a in ALGO_ORDER if a in st.session_state.results]
    all_algos = ["原图"] + sorted_algos

    # 【横向滚动布局】固定卡片宽度，超出时可左右滑动
    
    # 图像转 Base64
    def img_to_base64(img_rgb):
        """RGB 图像转 Base64"""
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')
    
    cards_html = '<div class="scroll-container">'
    
    for algo_name in all_algos:
        bg_color, text_color = algo_badge_map.get(algo_name, ("#e2e8f0", "#2d3748"))
        display_name = algo_name_map.get(algo_name, algo_name)
        
        if algo_name == "原图":
            # 原图卡片
            img_b64 = img_to_base64(st.session_state.img_rgb)
            cards_html += f'''
            <div class="image-card">
                <div class="card-header" style="background:linear-gradient(135deg,#f1f5f9 0%,#e2e8f0 100%);color:#475569;">{display_name}</div>
                <img src="data:image/jpeg;base64,{img_b64}" alt="{display_name}">
            </div>
            '''
        else:
            result_bgr = st.session_state.results.get(algo_name)
            if result_bgr is not None:
                result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
                img_b64 = img_to_base64(result_rgb)
                
                # 获取指标
                metrics = st.session_state.metrics.get(algo_name, {})
                rmse = metrics.get('RMSE', 'N/A')
                psnr = metrics.get('PSNR', 'N/A')
                rmse_display = f"{rmse:.2f}" if isinstance(rmse, (int, float)) else str(rmse)
                psnr_display = f"{psnr:.2f}" if isinstance(psnr, (int, float)) else str(psnr)
                
                cards_html += f'''
                <div class="image-card">
                    <div class="card-header" style="background:linear-gradient(135deg,{bg_color} 0%,{bg_color}dd 100%);color:{text_color};">{display_name}</div>
                    <img src="data:image/jpeg;base64,{img_b64}" alt="{display_name}">
                    <div class="metrics-row">
                        <div class="metric-box" style="background:linear-gradient(135deg,#f0f9ff 0%,#e0f2fe 100%);">
                            <div class="metric-label" style="color:#0369a1;">均方根误差 ↓</div>
                            <div class="metric-value" style="color:#0c4a6e;">{rmse_display}</div>
                        </div>
                        <div class="metric-box" style="background:linear-gradient(135deg,#f1f5f9 0%,#e2e8f0 100%);">
                            <div class="metric-label" style="color:#475569;">峰值信噪比 ↑</div>
                            <div class="metric-value" style="color:#334155;">{psnr_display}</div>
                        </div>
                    </div>
                </div>
                '''
    
    # 【新增】如果有参考图，放在最后展示
    if st.session_state.ref_bgr is not None:
        ref_rgb = cv2.cvtColor(st.session_state.ref_bgr, cv2.COLOR_BGR2RGB)
        ref_b64 = img_to_base64(ref_rgb)
        cards_html += f'''
        <div class="image-card">
            <div class="card-header" style="background:linear-gradient(135deg,#ecfdf5 0%,#d1fae5 100%);color:#065f46;">🎯 参考图 (Ground Truth)</div>
            <img src="data:image/jpeg;base64,{ref_b64}" alt="参考图">
        </div>
        '''
    
    cards_html += '</div>'
    
    # 使用 components.html 渲染，避免 st.markdown 对长 HTML 的限制
    import streamlit.components.v1 as components
    
    full_html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            * {{ margin: 0; padding: 0; box-sizing: border-box; }}
            body {{ font-family: 'Inter', -apple-system, sans-serif; background: transparent; }}
            .scroll-container {{
                display: flex;
                gap: 16px;
                overflow-x: auto;
                padding: 8px 4px 16px 4px;
                scroll-behavior: smooth;
            }}
            .scroll-container::-webkit-scrollbar {{ height: 8px; }}
            .scroll-container::-webkit-scrollbar-track {{ background: #f1f5f9; border-radius: 10px; }}
            .scroll-container::-webkit-scrollbar-thumb {{ background: linear-gradient(90deg, #6366f1, #8b5cf6); border-radius: 10px; }}
            .image-card {{
                flex: 0 0 280px;
                min-width: 280px;
                background: #fff;
                border-radius: 16px;
                overflow: hidden;
                box-shadow: 0 4px 20px rgba(0,0,0,0.05);
            }}
            .image-card img {{ width: 100%; height: auto; display: block; }}
            .card-header {{ padding: 10px 14px; font-weight: 600; font-size: 13px; letter-spacing: 0.3px; }}
            .metrics-row {{ display: flex; gap: 6px; padding: 6px 8px; background: #f8fafc; }}
            .metric-box {{ flex: 1; border-radius: 8px; padding: 6px 4px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.03); }}
            .metric-label {{ font-size: 10px; font-weight: 500; }}
            .metric-value {{ font-size: 15px; font-weight: 700; margin-top: 2px; }}
        </style>
    </head>
    <body>
        {cards_html}
    </body>
    </html>
    '''
    
    # 动态计算高度：根据图片比例调整
    components.html(full_html, height=320, scrolling=False)

    # 详细指标对比表格 - 无界化风格
    st.markdown("<p style='font-size:20px;font-weight:600;color:#2d3748;margin:12px 0 8px 0;'>详细评价指标</p>", unsafe_allow_html=True)

    metrics_data = []
    has_reference = st.session_state.ref_bgr is not None  # 检测是否有参考图
    
    for algo_name in sorted_algos:  # 使用排序后的算法列表
        m = st.session_state.metrics.get(algo_name, {})
        p = st.session_state.performance.get(algo_name, {})
        
        # 根据是否有参考图，决定指标显示方式
        rmse_val = m.get('RMSE', 'N/A')
        psnr_val = m.get('PSNR', 'N/A')
        ssim_val = m.get('SSIM', 'N/A')
        
        metrics_data.append({
            "算法": algo_name,
            "RMSE": rmse_val,
            "PSNR (dB)": psnr_val,
            "SSIM": ssim_val,
            "耗时 (ms)": p.get('time_ms', 0),
            "内存 (MB)": p.get('memory_mb', 0)
        })

    st.markdown("<style>div[data-testid='stHorizontalBlock'] {gap: 0.5rem;}</style>", unsafe_allow_html=True)

    # 【修复】调整表格高度公式，消除底部空行（表头 35px + 每行 35px）
    table_height = 35 + len(metrics_data) * 35
    
    # 动态列配置：根据是否有参考图决定指标列类型
    if has_reference:
        # 有参考图：使用 NumberColumn 格式化数字
        column_config = {
            "算法": st.column_config.TextColumn("算法", width="small"),
            "RMSE": st.column_config.NumberColumn("RMSE ↓", help="均方根误差，越小越好", format="%.2f"),
            "PSNR (dB)": st.column_config.NumberColumn("PSNR ↑", help="峰值信噪比，越大越好", format="%.2f"),
            "SSIM": st.column_config.NumberColumn("SSIM ↑", help="结构相似度，越接近1越好", format="%.4f"),
            "耗时 (ms)": st.column_config.NumberColumn("耗时", help="算法执行时间", format="%.0f ms"),
            "内存 (MB)": st.column_config.NumberColumn("内存", help="峰值内存使用", format="%.1f")
        }
    else:
        # 无参考图：指标列使用 TextColumn 兼容 "N/A"
        column_config = {
            "算法": st.column_config.TextColumn("算法", width="small"),
            "RMSE": st.column_config.TextColumn("RMSE ↓", help="均方根误差（无参考图时不可用）"),
            "PSNR (dB)": st.column_config.TextColumn("PSNR ↑", help="峰值信噪比（无参考图时不可用）"),
            "SSIM": st.column_config.TextColumn("SSIM ↑", help="结构相似度（无参考图时不可用）"),
            "耗时 (ms)": st.column_config.NumberColumn("耗时", help="算法执行时间", format="%.0f ms"),
            "内存 (MB)": st.column_config.NumberColumn("内存", help="峰值内存使用", format="%.1f")
        }

    st.dataframe(
        metrics_data,
        use_container_width=True,
        hide_index=True,
        height=table_height,
        column_config=column_config
    )

    # 操作按钮区域（仅在 result 状态显示）
    if st.session_state.ui_state == "result":
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        btn_cols = st.columns([1, 1, 2])

        # 【优化】"更换图片"按钮 - 切换到 changing 状态，而不是直接清空
        with btn_cols[0]:
            if st.button("更换图片", use_container_width=True):
                st.session_state.ui_state = "changing"
                st.rerun()

        # 下载结果按钮
        st.markdown("<p style='font-size:20px;font-weight:600;color:#2d3748;margin:12px 0 8px 0;'>下载结果</p>",
                    unsafe_allow_html=True)
        download_cols = st.columns(len(sorted_algos))
        for i, algo_name in enumerate(sorted_algos):  # 使用排序后的算法列表
            result_bgr = st.session_state.results[algo_name]
            with download_cols[i]:
                _, result_bytes = cv2.imencode('.png', result_bgr)
                st.download_button(
                    label=f"📥 {algo_name}",
                    data=result_bytes.tobytes(),
                    file_name=f"enhanced_{algo_name}_{st.session_state.uploaded_filename}",
                    mime="image/png",
                    use_container_width=True
                )

# ==========================================
# 9. 执行算法逻辑
# ==========================================
if execute_btn and st.session_state.img_bgr is not None:
    # 【修复】每次执行前强制清空旧结果，避免“脑缓存”问题
    st.session_state.results = {}
    st.session_state.metrics = {}
    st.session_state.performance = {}

    st.session_state.processing = True

    selected_algos = st.session_state.selected_algorithms
    img_bgr = st.session_state.img_bgr

    # 【优化】进度条放在主区域最上方
    with main_progress_placeholder.container():
        st.markdown("### ⏳ 正在处理...")
        progress_bar = st.progress(0)
        status_text = st.empty()

        params = {
            "he_params": st.session_state.he_params,
            "retinex_params": st.session_state.retinex_params
        }

        status_text.text(f"并行执行 {len(selected_algos)} 个算法...")

        # 并行执行所有算法（调用 image_processing 接口）
        results = image_processing.run_algorithms_parallel(img_bgr, selected_algos, params)

        # 处理结果
        for i, (algo_name, result_data) in enumerate(results.items()):
            progress = int((i + 1) / len(results) * 100)
            progress_bar.progress(progress)
            status_text.text(f"✅ 已完成: {algo_name} ({i+1}/{len(results)})")

            if result_data.get("success"):
                result_bgr = result_data["result"]
                st.session_state.results[algo_name] = result_bgr
                
                # 【条件化指标计算】有参考图才计算，否则返回 N/A
                if st.session_state.ref_bgr is not None:
                    # 有参考图：拿处理结果与参考图对比
                    _, ref_encoded = cv2.imencode('.png', st.session_state.ref_bgr)
                    _, result_encoded = cv2.imencode('.png', result_bgr)
                    metrics = calculate_metrics_cached(ref_encoded.tobytes(), result_encoded.tobytes())
                else:
                    # 无参考图：指标设为 N/A
                    metrics = {"RMSE": "N/A", "PSNR": "N/A", "SSIM": "N/A"}

                st.session_state.metrics[algo_name] = metrics
                st.session_state.performance[algo_name] = {
                    "time_ms": result_data["time_ms"],
                    "memory_mb": result_data["memory_mb"]
                }

        progress_bar.progress(100)
        status_text.text("✅ 所有算法执行完成！")
        time.sleep(0.5)  # 短暂显示完成状态

    main_progress_placeholder.empty()  # 清空进度区域

    st.session_state.processing = False
    st.session_state.ui_state = "result"
    st.rerun()
