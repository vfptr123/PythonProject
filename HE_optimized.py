"""
优化的直方图均衡化算法
使用向量化操作和缓存机制加速
"""
import cv2
import numpy as np

# 尝试导入 Numba，如不可用则使用纯 NumPy 实现
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    jit = lambda *args, **kwargs: lambda f: f  # 空装饰器
    prange = range


# ==========================================
# 优化1：全局直方图均衡化 (向量化版本)
# ==========================================
def global_histogram_equalization_optimized(img):
    """
    优化的全局直方图均衡化
    使用 OpenCV 内置优化 + 减少内存拷贝
    """
    # 直接操作 YUV 通道，避免不必要的拷贝
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    
    # 使用 OpenCV 的 equalizeHist (已经是高度优化的 C++ 实现)
    # 注意：equalizeHist 返回新数组，需要赋值回原通道
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


# ==========================================
# 优化3：CLAHE 算法 (缓存 CLAHE 对象)
# ==========================================
_clahe_cache = {}

def get_clahe(clip_limit=3.0, tile_grid_size=(8, 8)):
    """缓存 CLAHE 对象，避免重复创建"""
    key = (clip_limit, tile_grid_size)
    if key not in _clahe_cache:
        _clahe_cache[key] = cv2.createCLAHE(
            clipLimit=clip_limit, 
            tileGridSize=tile_grid_size
        )
    return _clahe_cache[key]


def clahe_enhancement_optimized(img, clip_limit=3.0, tile_grid_size=(8, 8)):
    """
    优化的 CLAHE 算法
    使用缓存的 CLAHE 对象
    """
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    
    # 获取缓存的 CLAHE 对象
    clahe = get_clahe(clip_limit, tile_grid_size)
    
    # 应用 CLAHE，返回结果赋值回原通道
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


# ==========================================
# 优化4：批量处理版本 (用于并行执行多个图像)
# ==========================================
def batch_he_process(images, method='global'):
    """
    批量处理多张图像
    利用 CPU 缓存局部性提升性能
    """
    results = []
    for img in images:
        if method == 'global':
            results.append(global_histogram_equalization_optimized(img))
        elif method == 'clahe':
            results.append(clahe_enhancement_optimized(img))
    return results


# ==========================================
# 保持向后兼容的别名
# ==========================================
global_histogram_equalization = global_histogram_equalization_optimized
clahe_enhancement = clahe_enhancement_optimized

# 如果 Numba 可用，添加 JIT 编译版本
if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, cache=True)
    def _compute_histogram_numba(channel, hist_size=256):
        """使用 Numba 并行计算直方图"""
        hist = np.zeros(hist_size, dtype=np.int32)
        h, w = channel.shape
        for i in prange(h):
            for j in range(w):
                hist[channel[i, j]] += 1
        return hist
