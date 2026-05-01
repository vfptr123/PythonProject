"""
低照度图像增强评估系统 v5.1
===============================
职责：纯 UI 层 — 页面渲染、用户交互、状态管理
业务逻辑委托 image_processing.py
"""
import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import base64
import threading
import image_processing
from styles import GLOBAL_CSS, SCROLL_LAYOUT_CSS

# 后台预加载 NIQE（首次 import pyiqa ~3s，用户上传图片期间即完成）
threading.Thread(target=image_processing._get_niqe_model, daemon=True).start()

# ==========================================
# 1. 页面配置（必须是第一个 Streamlit 命令）
# ==========================================
st.set_page_config(
    page_title="低照度图像增强评估系统",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ==========================================
# 2. 常量
# ==========================================
ALGO_ORDER = ["HE", "MSRCR", "Zero-DCE"]

# ALGO_INFO = {
#     "HE":       ("HE",       "直方图均衡化"),
#     "MSRCR":  ("MSRCR",  "带色彩恢复的多尺度Retinex"),
#     "Zero-DCE": ("Zero-DCE", "零参考深度学习"),
# }

ALGO_INFO = {
    "HE":       ("HE",       "直方图均衡化"),
    "MSRCR":  ("MSRCR",  "带色彩恢复的多尺度Retinex"),
    "Zero-DCE": ("Zero-DCE", "零参考深度学习"),
}

ALGO_BADGE = {
    "HE":       ("#ebf8ff", "#2b6cb0"),
    "MSRCR":  ("#faf5ff", "#553c9a"),
    "Zero-DCE": ("#fff5f5", "#c53030"),
}

ALGO_DISPLAY = {
    "原图":     "原图",
    "HE":       "HE",
    "MSRCR":  "MSRCR",
    "Zero-DCE": "Zero-DCE",
}

# 指标卡片配置：(label, 方向箭头, 渐变色, 标签色, 值色, format)
METRIC_CARDS = [
    ("峰值信噪比", "↑", "#e6f4ff", "#d6eeff", "#0369a1", "#0c4a6e", "%.2f"),
    ("结构相似度", "↑", "#f0e6ff", "#e6d9ff", "#6b47b8", "#4c1d95", "%.4f"),
    ("自然质量评价", "↓", "#e0e7ff", "#d5defa", "#4b5e8a", "#334155", "%.2f"),
]

# ==========================================
# 3. Session State 初始化
# ==========================================
_DEFAULTS = {
    "ui_state": "upload",
    "selected_algorithms": [],
    "processing": False,
    "results": {}, "metrics": {}, "performance": {},
    "img_bgr": None, "img_rgb": None, "uploaded_filename": "", "file_bytes": None,
    "ref_bgr": None, "ref_rgb": None, "ref_filename": "", "ref_file_bytes": None,
    "he_params": {"clip_limit": 3.0, "tile_grid": 8},
    "MSRCR_params": {
        "sigma_list": "15,80,250",
        "G": 5.0, "b": 25.0, "alpha": 125.0, "beta": 46.0,
        "low_clip": 0.01, "high_clip": 0.99,
    },
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ==========================================
# 4. 工具函数
# ==========================================

# key 映射：主图 vs 参考图
_IMG_KEYS = {
    "": ("img_bgr", "img_rgb", "uploaded_filename", "file_bytes"),
    "ref_": ("ref_bgr", "ref_rgb", "ref_filename", "ref_file_bytes"),
}

def _decode_image(uploaded_file, prefix: str) -> bool:
    """通用图片解码 → session_state。prefix: '' (低照度) / 'ref_' (参考图)"""
    if uploaded_file is None:
        return False
    file_bytes = uploaded_file.getvalue()
    bgr = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return False
    k_bgr, k_rgb, k_name, k_bytes = _IMG_KEYS[prefix]
    st.session_state[k_bgr] = bgr
    st.session_state[k_rgb] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    st.session_state[k_name] = uploaded_file.name
    st.session_state[k_bytes] = file_bytes
    return True


def clear_results():
    """清空算法结果缓存"""
    for k in ("results", "metrics", "performance"):
        st.session_state[k] = {}


def clear_ref():
    """清空参考图缓存"""
    for k in ("ref_bgr", "ref_rgb", "ref_filename", "ref_file_bytes"):
        st.session_state[k] = None if k != "ref_filename" else ""


def fmt(value, pattern: str) -> str:
    """安全格式化指标值（处理 nan / N/A / 非数字）"""
    if isinstance(value, (int, float)) and not np.isnan(value):
        return pattern % value
    return str(value)


def img_to_b64(img_rgb: np.ndarray) -> str:
    """RGB ndarray → JPEG Base64"""
    _, buf = cv2.imencode('.jpg', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                          [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


def _build_metric_html(psnr, ssim, niqe) -> str:
    """生成单张卡片底部三栏指标 HTML"""
    values = [psnr, ssim, niqe]
    html = '<div class="metrics-row">'
    for (label, arrow, bg1, bg2, lbl_color, val_color, pattern), val in zip(METRIC_CARDS, values):
        html += (
            f'<div class="metric-box" style="background:linear-gradient(135deg,{bg1} 0%,{bg2} 100%);">'
            f'<div class="metric-label" style="color:{lbl_color};">{label} {arrow}</div>'
            f'<div class="metric-value" style="color:{val_color};">{fmt(val, pattern)}</div>'
            f'</div>'
        )
    html += '</div>'
    return html


def build_cards_html() -> str:
    """构建横向滚动卡片 HTML（原图 + 各算法结果 + 参考图）"""
    s = st.session_state
    sorted_algos = [a for a in ALGO_ORDER if a in s.results]
    cards = '<div class="scroll-container">'

    # 原图
    cards += (
        f'<div class="image-card">'
        f'<div class="card-header" style="background:linear-gradient(135deg,#f1f5f9,#e2e8f0);color:#475569;text-align:center;">原图</div>'
        f'<img src="data:image/jpeg;base64,{img_to_b64(s.img_rgb)}" alt="原图">'
        f'</div>'
    )

    # 算法结果
    for algo in sorted_algos:
        bg, tc = ALGO_BADGE.get(algo, ("#e2e8f0", "#2d3748"))
        name = ALGO_DISPLAY.get(algo, algo)
        result_bgr = s.results[algo]
        result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
        m = s.metrics.get(algo, {})

        cards += (
            f'<div class="image-card">'
            f'<div class="card-header" style="background:linear-gradient(135deg,{bg} 0%,{bg}dd 100%);color:{tc};text-align:center;">{name}</div>'
            f'<img src="data:image/jpeg;base64,{img_to_b64(result_rgb)}" alt="{name}">'
            f'{_build_metric_html(m.get("PSNR", "N/A"), m.get("SSIM", "N/A"), m.get("NIQE", "N/A"))}'
            f'</div>'
        )

    # 参考图（可选）
    if s.ref_bgr is not None:
        ref_rgb = cv2.cvtColor(s.ref_bgr, cv2.COLOR_BGR2RGB)
        cards += (
            f'<div class="image-card">'
            f'<div class="card-header" style="background:linear-gradient(135deg,#ecfdf5,#d1fae5);color:#065f46;">参考图</div>'
            f'<img src="data:image/jpeg;base64,{img_to_b64(ref_rgb)}" alt="参考图">'
            f'</div>'
        )

    cards += '</div>'
    return cards


# ==========================================
# 5. 文件上传区域
# ==========================================
main_progress_placeholder = st.empty()

if st.session_state.ui_state == "upload":
    st.markdown("## 低照度图像增强评估系统")

    st.markdown("##### 低照度图像（必选）")
    st.caption("上传图片并选择算法对比增强效果")
    up_file = st.file_uploader("选择低照度图像", type=['png', 'jpg', 'jpeg', 'bmp'],
                               label_visibility="collapsed", key="uploader_state_a")
    _decode_image(up_file, "")

    st.markdown("##### 参考图像（可选）")
    st.caption("用于计算 PSNR/SSIM")
    ref_file = st.file_uploader("选择参考图像", type=['png', 'jpg', 'jpeg', 'bmp'],
                                label_visibility="collapsed", key="uploader_ref_a")
    if ref_file is not None:
        _decode_image(ref_file, "ref_")

elif st.session_state.ui_state == "changing":
    st.markdown("## 更换图片")
    st.info("请选择新的图片文件。上传成功后，旧的对比结果将被清空。")

    st.markdown("##### 低照度图像（必选）")
    new_file = st.file_uploader("选择新图像", type=['png', 'jpg', 'jpeg', 'bmp'],
                                label_visibility="collapsed", key="uploader_change")

    st.markdown("##### 参考图像（可选）")
    new_ref = st.file_uploader("选择参考图像", type=['png', 'jpg', 'jpeg', 'bmp'],
                               label_visibility="collapsed", key="uploader_ref_change")

    col_cancel, _ = st.columns([1, 3])
    with col_cancel:
        if st.button("❌ 取消更换", use_container_width=True):
            st.session_state.ui_state = "result"
            st.rerun()

    if new_file is not None and _decode_image(new_file, ""):
        clear_results()
        if new_ref is not None:
            _decode_image(new_ref, "ref_")
        else:
            clear_ref()
        st.session_state.ui_state = "upload"
        st.rerun()

    st.markdown("---")

# ==========================================
# 6. 侧边栏
# ==========================================
with st.sidebar:
    st.markdown("## 算法控制台")
    st.markdown("<hr style='margin:8px 0;border:none;border-top:1px solid #e2e8f0;'>", unsafe_allow_html=True)
    st.markdown("### 选择算法（可多选）")

    selected = []
    for algo, (label, desc) in ALGO_INFO.items():
        if st.checkbox(f"{label} - {desc}", value=algo in st.session_state.selected_algorithms, key=f"chk_{algo}"):
            selected.append(algo)
    st.session_state.selected_algorithms = selected

    # HE 参数
    if "HE" in selected:
        with st.expander("HE 参数", expanded=False):
            st.caption("HE 为全局均衡，无需额外参数")

    # MSRCR 参数
    if "MSRCR" in selected:
        with st.expander("MSRCR 参数", expanded=False):
            p = st.session_state.MSRCR_params
            p["sigma_list"] = st.text_input(
                "Sigma List", p["sigma_list"], placeholder="例: 15,80,250",
                help="高斯模糊尺度列表（英文逗号隔开）\n\n小尺度：保留细节\n\n中尺度：平衡细节与光照\n\n大尺度：估计全局光照\n\n📌 参考值：15,80,250"
            )
            cg, cb = st.columns(2)
            with cg:
                p["G"] = st.number_input("G", 1.0, 20.0, p["G"], 0.5, format="%.1f",
                                         help="增益系数\n⬆️ 调大：图像整体变亮\n📌 参考值：5.0")
            with cb:
                p["b"] = st.number_input("b", 1.0, 100.0, p["b"], 1.0, format="%.1f",
                                         help="偏移量\n⬆️ 调大：暗部提亮更明显\n📌 参考值：25.0")
            ca, cbt = st.columns(2)
            with ca:
                p["alpha"] = st.number_input("α", 50.0, 300.0, p["alpha"], 5.0, format="%.0f",
                                             help="颜色恢复强度\n⬆️ 调大：色彩更鲜艳饱和\n📌 参考值：125")
            with cbt:
                p["beta"] = st.number_input("β", 10.0, 100.0, p["beta"], 1.0, format="%.0f",
                                            help="颜色恢复偏移\n配合 α 微调色彩平衡\n📌 参考值：46")
            clo, chi = st.columns(2)
            with clo:
                p["low_clip"] = st.number_input("暗部截断", 0.0, 0.1, p["low_clip"], 0.01, format="%.2f",
                                                help="暗部截断比例\n⬆️ 调大：去除更多暗部噪点\n📌 参考值：0.01")
            with chi:
                p["high_clip"] = st.number_input("亮部截断", 0.9, 1.0, p["high_clip"], 0.01, format="%.2f",
                                                 help="亮部截断比例\n⬇️ 调小：抑制过曝区域\n📌 参考值：0.99")

    if "Zero-DCE" in selected:
        with st.expander("深度学习说明", expanded=False):
            st.caption("首次加载需要 5-15 秒，之后会缓存")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    has_image = st.session_state.img_bgr is not None
    has_algo = len(selected) > 0
    can_execute = has_image and has_algo and not st.session_state.processing

    if not has_image:
        st.caption("⚠️ 请先上传图片")
    elif not has_algo:
        st.caption("⚠️ 请至少选择一个算法")

    execute_btn = st.button("🚀 执行选中算法", type="primary",
                            use_container_width=True, disabled=not can_execute)

# ==========================================
# 7. 主区域渲染
# ==========================================

# --- 上传状态 ---
if st.session_state.ui_state == "upload":
    if st.session_state.img_bgr is not None:
        h, w = st.session_state.img_bgr.shape[:2]
        kb = len(st.session_state.file_bytes) / 1024 if st.session_state.file_bytes else 0
        st.success(f"✅ **低照度图片已就绪**  |  `{st.session_state.uploaded_filename}`  |  {w}×{h} px  |  {kb:.1f} KB")

        if st.session_state.ref_bgr is not None:
            rh, rw = st.session_state.ref_bgr.shape[:2]
            rkb = len(st.session_state.ref_file_bytes) / 1024 if st.session_state.ref_file_bytes else 0
            st.info(f"✅ **参考图已就绪**  |  `{st.session_state.ref_filename}`  |  {rw}×{rh} px  |  {rkb:.1f} KB")
        else:
            st.caption("⚠️ 未上传参考图，PSNR/SSIM 指标将显示为 N/A")

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

# --- 结果 / 更换状态 ---
elif st.session_state.ui_state in ("result", "changing"):
    title = "多算法对比结果" if st.session_state.ui_state == "result" else "当前对比结果（即将被替换）"
    st.markdown(f"<p style='font-size:20px;font-weight:600;color:#2d3748;margin:0 0 8px 0;'>{title}</p>", unsafe_allow_html=True)

    # 横向卡片
    full_html = f'<!DOCTYPE html><html><head><style>{SCROLL_LAYOUT_CSS}</style></head><body>{build_cards_html()}</body></html>'
    components.html(full_html, height=320, scrolling=False)

    # 详细指标表格
    st.markdown("<p style='font-size:20px;font-weight:600;color:#2d3748;margin:12px 0 8px 0;'>详细评价指标</p>", unsafe_allow_html=True)

    sorted_algos = [a for a in ALGO_ORDER if a in st.session_state.results]
    has_ref = st.session_state.ref_bgr is not None
    rows = []
    for algo in sorted_algos:
        m = st.session_state.metrics.get(algo, {})
        p = st.session_state.performance.get(algo, {})
        rows.append({
            "算法": algo,
            "PSNR (dB)": m.get("PSNR", "N/A"),
            "SSIM": m.get("SSIM", "N/A"),
            "NIQE": m.get("NIQE", "N/A"),
            "耗时 (ms)": p.get("time_ms", 0),
            "内存 (MB)": p.get("memory_mb", 0),
        })

    # 动态列配置：有参考图用 Number，无参考图用 Text（N/A 显示友好）
    psnr_col = st.column_config.NumberColumn("PSNR ↑", help="峰值信噪比，越大越好", format="%.2f") if has_ref \
        else st.column_config.TextColumn("PSNR ↑", help="峰值信噪比（无参考图时不可用）")
    ssim_col = st.column_config.NumberColumn("SSIM ↑", help="结构相似度，越接近1越好", format="%.4f") if has_ref \
        else st.column_config.TextColumn("SSIM ↑", help="结构相似度（无参考图时不可用）")

    st.markdown("<style>div[data-testid='stHorizontalBlock'] {gap: 0.5rem;}</style>", unsafe_allow_html=True)
    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
        height=35 + len(rows) * 35,
        column_config={
            "算法": st.column_config.TextColumn("算法", width="small"),
            "PSNR (dB)": psnr_col,
            "SSIM": ssim_col,
            "NIQE": st.column_config.NumberColumn("NIQE ↓", help="自然图像质量评价，越小越好", format="%.2f"),
            "耗时 (ms)": st.column_config.NumberColumn("耗时", help="算法执行时间", format="%.0f ms"),
            "内存 (MB)": st.column_config.NumberColumn("内存", help="峰值内存使用", format="%.1f"),
        },
    )

    # 操作 & 下载
    if st.session_state.ui_state == "result":
        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        if st.button("更换图片", use_container_width=False):
            st.session_state.ui_state = "changing"
            st.rerun()

        st.markdown("<p style='font-size:20px;font-weight:600;color:#2d3748;margin:12px 0 8px 0;'>下载结果</p>", unsafe_allow_html=True)
        dl_cols = st.columns(len(sorted_algos))
        for i, algo in enumerate(sorted_algos):
            with dl_cols[i]:
                _, png = cv2.imencode('.png', st.session_state.results[algo])
                st.download_button(
                    f"📥 {algo}", png.tobytes(),
                    file_name=f"enhanced_{algo}_{st.session_state.uploaded_filename}",
                    mime="image/png", use_container_width=True,
                )

# ==========================================
# 8. 执行算法
# ==========================================
if execute_btn and st.session_state.img_bgr is not None:
    clear_results()
    st.session_state.processing = True
    algos = st.session_state.selected_algorithms
    img = st.session_state.img_bgr

    with main_progress_placeholder.container():
        st.markdown("### ⏳ 正在处理...")
        bar = st.progress(0)
        status = st.empty()

        params = {
            "he_params": st.session_state.he_params,
            "MSRCR_params": st.session_state.MSRCR_params,
        }
        status.text(f"并行执行 {len(algos)} 个算法...")
        results = image_processing.run_algorithms_parallel(img, algos, params)

        total = len(results) * 2
        step = 0
        for name, data in results.items():
            if not data.get("success"):
                continue
            result_bgr = data["result"]
            st.session_state.results[name] = result_bgr

            step += 1
            bar.progress(int(step / total * 100))
            status.text(f"计算 {name} 指标...")

            niqe = image_processing.calculate_niqe(result_bgr)
            metrics = image_processing.calculate_metrics(st.session_state.ref_bgr, result_bgr) \
                if st.session_state.ref_bgr is not None else {"PSNR": "N/A", "SSIM": "N/A"}
            metrics["NIQE"] = niqe

            st.session_state.metrics[name] = metrics
            st.session_state.performance[name] = {
                "time_ms": data["time_ms"], "memory_mb": data["memory_mb"],
            }

            step += 1
            bar.progress(int(step / total * 100))
            status.text(f"✅ {name} 完成 ({step // 2}/{len(results)})")

        bar.progress(100)
        status.text("✅ 所有算法执行完成！")

    main_progress_placeholder.empty()
    st.session_state.processing = False
    st.session_state.ui_state = "result"
    st.rerun()
