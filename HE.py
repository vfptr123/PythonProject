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