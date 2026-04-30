"""
图像处理算法调度中心 v3.0
================================
职责划分：
1. 算法实现：HE/Retinex/Zero-DCE
2. 指标计算：RMSE/PSNR/SSIM
3. 性能监控：耗时/内存
4. 算法调度：并行执行

app.py 只负责 UI 展示，调用此文件提供的接口
"""
import sys
import os
import cv2
import numpy as np
import torch
import time
import psutil
from PIL import Image
import torchvision.transforms as transforms
import streamlit as st  # 引入 streamlit 用于缓存装饰器
from concurrent.futures import ThreadPoolExecutor, as_completed
from skimage.metrics import structural_similarity as ssim_metric

# ==========================================
# 配置 Zero-DCE 环境
# ==========================================
zerodce_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Zero-DCE_code')
if zerodce_path not in sys.path:
    sys.path.append(zerodce_path)

try:
    from model import enhance_net_nopool
except ImportError as e:
    print(f"Warning: Zero-DCE modules not found: {e}")

# 导入优化后的算法
try:
    import HE_optimized as HE
    import Retinex_optimized as Retinex
    print("使用优化版本的算法")
except ImportError:
    import HE
    import Retinex
    print("使用标准版本的算法")

# ==========================================
# 传统算法部分 (全部使用优化版本)
# ==========================================

def process_with_he(img_bgr):
    """调用优化后的 HE (全局直方图均衡化) 算法"""
    return HE.global_histogram_equalization_optimized(img_bgr)

def process_with_retinex(img_bgr, sigma_list=None, G=5.0, b=25.0, 
                         alpha=125.0, beta=46.0, low_clip=0.01, high_clip=0.99):
    """调用优化后的 Retinex (MSRCR) 算法"""
    if sigma_list is None:
        sigma_list = [15, 80, 250]
    return Retinex.MSRCR_optimized(
        img_bgr, 
        sigma_list=sigma_list, 
        G=G, b=b, 
        alpha=alpha, beta=beta,
        low_clip=low_clip, high_clip=high_clip
    )

# ==========================================
# Zero-DCE 深度学习算法部分
# ==========================================

@st.cache_resource(show_spinner="🔄 首次加载 Zero-DCE 模型...")  # 缓存模型，避免每次 rerun 重新加载
def init_zerodce_model():
    """
    初始化 Zero-DCE 模型。
    使用 @st.cache_resource 确保模型仅在首次调用时加载。
    """
    try:
        print("正在加载 Zero-DCE 模型...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        DCE_net = enhance_net_nopool().to(device)
        
        # 加载预训练权重
        snapshot_path = os.path.join(zerodce_path, 'snapshots', 'Epoch199.pth')
        if os.path.exists(snapshot_path):
            DCE_net.load_state_dict(torch.load(snapshot_path, map_location=device))
            print(f"Zero-DCE 模型加载成功！({snapshot_path})")
        else:
            print(f"Warning: 未找到预训练权重: {snapshot_path}")
            return None
            
        DCE_net.eval()
        return DCE_net
    except Exception as e:
        print(f"Zero-DCE 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_with_zerodce(model, img_bgr):
    """
    使用加载好的 Zero-DCE 模型处理图片
    """
    if model is None:
        print("Error: Zero-DCE Model is None")
        return img_bgr

    try:
        device = next(model.parameters()).device
        
        # BGR -> RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 归一化到 [0, 1]
        img_float = img_rgb.astype(np.float32) / 255.0
        
        # 转换为 Tensor [H, W, C] -> [C, H, W]
        img_tensor = torch.from_numpy(img_float).permute(2, 0, 1).unsqueeze(0).to(device)
        
        # 推理
        with torch.no_grad():
            _, enhanced_image, _ = model(img_tensor)
        
        # 转换回 numpy
        enhanced_np = enhanced_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        enhanced_np = np.clip(enhanced_np * 255, 0, 255).astype(np.uint8)
        
        # RGB -> BGR
        result_bgr = cv2.cvtColor(enhanced_np, cv2.COLOR_RGB2BGR)
        return result_bgr
        
    except Exception as e:
        print(f"Zero-DCE 推理出错: {e}")
        import traceback
        traceback.print_exc()
        return img_bgr


# ==========================================
# 指标计算模块
# ==========================================

def calculate_metrics(img1_bgr: np.ndarray, img2_bgr: np.ndarray) -> dict:
    """
    计算两张图像的质量指标
    
    Args:
        img1_bgr: 参考图像 (BGR 格式)
        img2_bgr: 处理后图像 (BGR 格式)
    
    Returns:
        dict: 包含 RMSE, PSNR, SSIM 的字典
    """
    # 计算 RMSE
    diff = img1_bgr.astype(np.float32) - img2_bgr.astype(np.float32)
    mse = np.mean(diff * diff)
    rmse = np.sqrt(mse)
    
    # 计算 PSNR
    psnr = 20 * np.log10(255.0 / rmse) if mse > 0 else 99.0
    
    # 计算 SSIM（缩小图像加速）
    scale = 0.5
    h, w = img1_bgr.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    img1_gray = cv2.resize(cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY), (new_w, new_h))
    img2_gray = cv2.resize(cv2.cvtColor(img2_bgr, cv2.COLOR_BGR2GRAY), (new_w, new_h))
    ssim_score = ssim_metric(img1_gray, img2_gray, data_range=255)
    
    return {
        "RMSE": float(rmse),
        "PSNR": float(psnr),
        "SSIM": float(ssim_score)
    }


def calculate_metrics_from_bytes(img1_bytes: bytes, img2_bytes: bytes) -> dict:
    """
    从字节流计算指标（用于 Streamlit 缓存）
    """
    img1 = cv2.imdecode(np.frombuffer(img1_bytes, np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.frombuffer(img2_bytes, np.uint8), cv2.IMREAD_COLOR)
    return calculate_metrics(img1, img2)


# ==========================================
# 性能监控模块
# ==========================================

def get_memory_usage() -> float:
    """
    获取当前进程的内存使用量 (MB)
    """
    return psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024


# ==========================================
# 算法调度模块
# ==========================================

def process_algorithm(algo_name: str, img_bgr: np.ndarray, params: dict) -> dict:
    """
    执行单个算法并记录性能指标
    
    Args:
        algo_name: 算法名称 (HE/Retinex/Zero-DCE)
        img_bgr: 输入图像 (BGR 格式)
        params: 算法参数字典
    
    Returns:
        dict: 包含 result, time_ms, memory_mb, success 的字典
    """
    start_time = time.time()
    initial_memory = get_memory_usage()
    
    try:
        if algo_name == "HE":
            result = process_with_he(img_bgr)
        elif algo_name == "Retinex":
            retinex_params = params.get("retinex_params", {})
            sigma_str = retinex_params.get("sigma_list", "15,80,250")
            sigma_list = [float(x.strip()) for x in sigma_str.split(",")]
            result = process_with_retinex(
                img_bgr,
                sigma_list=sigma_list,
                G=retinex_params.get("G", 5.0),
                b=retinex_params.get("b", 25.0),
                alpha=retinex_params.get("alpha", 125.0),
                beta=retinex_params.get("beta", 46.0),
                low_clip=retinex_params.get("low_clip", 0.01),
                high_clip=retinex_params.get("high_clip", 0.99)
            )
        elif algo_name == "Zero-DCE":
            model = init_zerodce_model()
            result = process_with_zerodce(model, img_bgr) if model else img_bgr
        else:
            result = img_bgr
        
        execution_time = (time.time() - start_time) * 1000
        memory_used = max(0, get_memory_usage() - initial_memory)
        
        return {
            "result": result,
            "time_ms": execution_time,
            "memory_mb": memory_used,
            "success": True
        }
    except Exception as e:
        return {
            "error": str(e),
            "success": False
        }


def _process_algorithm_wrapper(args):
    """
    并行执行的包装函数
    """
    algo_name, img_bgr, params = args
    return algo_name, process_algorithm(algo_name, img_bgr, params)


def run_algorithms_parallel(img_bgr: np.ndarray, algorithms: list, params: dict, max_workers: int = 3) -> dict:
    """
    并行执行多个算法
    
    Args:
        img_bgr: 输入图像 (BGR 格式)
        algorithms: 算法名称列表
        params: 算法参数字典
        max_workers: 最大并行数
    
    Returns:
        dict: {algo_name: result_dict} 的字典
    """
    args_list = [(algo, img_bgr, params) for algo in algorithms]
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_algorithm_wrapper, args): args[0] for args in args_list}
        for future in as_completed(futures):
            algo_name, result = future.result()
            results[algo_name] = result
    
    return results
