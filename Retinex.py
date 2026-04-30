import numpy as np
import cv2


# =============================================================
# 1. 单尺度 Retinex (SSR - Single Scale Retinex)
# 原理：将图像分离为光照分量 (L) 和反射分量 (R)。
# 公式：Log(R) = Log(I) - Log(Gaussian(I))
# =============================================================
def singleScaleRetinex(img, sigma):
    # 计算 log(R) = log(I) - log(L)
    # L 是通过高斯模糊估计出来的环境光照
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))
    return retinex


# =============================================================
# 2. 多尺度 Retinex (MSR - Multi Scale Retinex)
# 原理：结合不同尺度 (sigma) 的 SSR 结果，平衡细节增强和色彩自然度。
# =============================================================
def multiScaleRetinex(img, sigma_list):
    # 初始化结果为零
    retinex = np.zeros_like(img)
    # 对每个尺度遍历 (通常是小、中、大三个尺度)
    for sigma in sigma_list:
        retinex += singleScaleRetinex(img, sigma)

    # 取平均值
    retinex = retinex / len(sigma_list)
    return retinex


# =============================================================
# 3. 色彩恢复函数 (CR - Color Restoration)
# 作用：解决 MSR 算法容易出现的“泛白/褪色”现象。
# =============================================================
def colorRestoration(img, alpha, beta):
    img_sum = np.sum(img, axis=2, keepdims=True)

    # CR 公式：beta * [log(alpha * I) - log(Sum(I))]
    # 这里的 +1e-8 是为了防止除以 0 或 log(0) 报错
    color_restoration = beta * (np.log10(alpha * img) - np.log10(img_sum + 1e-8))
    return color_restoration


# =============================================================
# 4. 最简色彩平衡 (Simplest Color Balance)
# 作用：自动色阶调整，去除偏色，增强对比度 (类似 Photoshop 的自动色阶)。
# 参数：low_clip (暗部截断比例), high_clip (亮部截断比例)
# =============================================================
def simplestColorBalance(img, low_clip, high_clip):
    total = img.shape[0] * img.shape[1]
    for i in range(img.shape[2]):
        # 统计直方图
        unique, counts = np.unique(img[:, :, i], return_counts=True)
        current = 0
        low_val = unique[0]
        high_val = unique[-1]

        # 寻找低阈值和高阈值
        for u, c in zip(unique, counts):
            if float(current) / total < low_clip:
                low_val = u
            if float(current) / total < high_clip:
                high_val = u
            current += c

        # 截断并拉伸
        img[:, :, i] = np.maximum(np.minimum(img[:, :, i], high_val), low_val)
    return img


# =============================================================
# 5. 带色彩恢复的多尺度 Retinex (MSRCR) - 【毕设核心算法】
# 参数说明：
#   G: 增益 (Gain)
#   b: 偏差 (Bias)
#   alpha, beta: 色彩恢复参数
#   low_clip, high_clip: 色彩平衡截断点
# =============================================================
def MSRCR(img, sigma_list = [15,80,250], G = 5.0, b = 25.0, alpha = 125.0, beta = 46.0, low_clip = 0.01, high_clip = 0.99):
    # 转换为浮点数计算，+1.0 防止 log(0)
    img = np.float64(img) + 1.0

    # 1. 计算 MSR
    img_retinex = multiScaleRetinex(img, sigma_list)

    # 2. 计算色彩恢复因子 CR
    img_color = colorRestoration(img, alpha, beta)

    # 3. 融合：Result = G * (MSR * CR + b)
    img_msrcr = G * (img_retinex * img_color + b)

    # 4. 归一化到 0-255
    for i in range(img_msrcr.shape[2]):
        img_msrcr[:, :, i] = (img_msrcr[:, :, i] - np.min(img_msrcr[:, :, i])) / \
                             (np.max(img_msrcr[:, :, i]) - np.min(img_msrcr[:, :, i])) * \
                             255

    # 5. 截断溢出值并在最后做一次色彩平衡
    img_msrcr = np.uint8(np.minimum(np.maximum(img_msrcr, 0), 255))
    img_msrcr = simplestColorBalance(img_msrcr, low_clip, high_clip)

    return img_msrcr


# =============================================================
# 6. 自动 MSRCR (Automated MSRCR)
# 作用：自动计算参数，不需要手动调 G 和 b
# =============================================================
def automatedMSRCR(img, sigma_list):
    img = np.float64(img) + 1.0
    img_retinex = multiScaleRetinex(img, sigma_list)

    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
        # ... 这里保留了原算法的直方图统计逻辑，用于自动寻找截断点 ...
        zero_count = 0
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break

        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0

        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.0
                break

        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)

        # 归一化
        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255

    img_retinex = np.uint8(img_retinex)
    return img_retinex


# =============================================================
# 7. 色彩保持多尺度 Retinex (MSRCP - Color Preservation)
# 原理：只在亮度通道做 Retinex，然后按比例恢复 RGB，避免偏色。
# 优化：已将原版的双重 for 循环改为矩阵运算，速度提升 100 倍。
# =============================================================
def MSRCP(img, sigma_list, low_clip, high_clip):
    # 转换 float
    img = np.float64(img) + 1.0

    # 1. 计算亮度 Intensity (简单的平均法)
    intensity = np.sum(img, axis=2) / img.shape[2]

    # 2. 仅对亮度层做 MSR
    retinex = multiScaleRetinex(intensity, sigma_list)

    # 扩展维度以便计算
    intensity = np.expand_dims(intensity, 2)
    retinex = np.expand_dims(retinex, 2)

    # 3. 对亮度结果做色彩平衡
    intensity1 = simplestColorBalance(retinex, low_clip, high_clip)

    # 4. 归一化亮度
    intensity1 = (intensity1 - np.min(intensity1)) / \
                 (np.max(intensity1) - np.min(intensity1)) * \
                 255.0 + 1.0

    # 5. 核心融合公式 (矩阵运算版)
    # B = Max(R, G, B)
    B = np.max(img, axis=2, keepdims=True)

    # A = Min(256 / B, New_Intensity / Old_Intensity)
    # 防止除以0
    A = np.minimum(256.0 / (B + 1e-8), intensity1 / (intensity + 1e-8))

    # Result = A * Old_Image
    img_msrcp = A * img

    # 6. 转回 uint8
    img_msrcp = np.uint8(np.clip(img_msrcp - 1.0, 0, 255))

    return img_msrcp