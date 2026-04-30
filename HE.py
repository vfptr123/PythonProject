import cv2


# ==========================================
# 算法1：普通全局直方图均衡化 (Global HE)
# 缺点：容易产生噪点，亮的地方过曝
# ==========================================
def global_histogram_equalization(img):
    # 1. 【核心步骤】颜色空间转换：BGR -> YUV
    # 理论对应：不直接处理 RGB，而是把“亮度(Y)”和“颜色(UV)”剥离开
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # 2. 提取 Y 通道 (亮度层)
    # img_yuv[:,:,0] 代表第0层，也就是 Y 层
    # equalizeHist 是 OpenCV 自带的“把直方图拉平”的函数
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # 3. 【合并】把处理好的 Y 层和原来的 UV 层拼回去，转回 BGR
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return img_output


# ==========================================
# 算法2：限制对比度自适应直方图均衡化 (CLAHE)
# 优点：毕设推荐算法！控制噪点，细节更自然
# ==========================================
def clahe_enhancement(img):
    # 1. 同样先转到 YUV
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # 2. 创建 CLAHE 对象 (这就是那个“把操场切成小块”的班主任)
    # clipLimit=3.0: 限制阈值。数值越大越亮，但噪点越多。一般 2.0-4.0 之间。
    # tileGridSize=(8,8): 把图像切成 8x8 的小方块进行局部处理。
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

    # 3. 只对 Y 通道应用 CLAHE
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])

    # 4. 转回 BGR
    img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    return img_output