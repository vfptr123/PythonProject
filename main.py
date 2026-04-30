import cv2
import os
import matplotlib.pyplot as plt
import HE
import Retinex

if __name__ == "__main__":
    # 1. 配置路径
    input_img_name = '1.png'  # 输入图片名
    output_folder = 'Results_Images'  # 想要保存到的文件夹名字

    # 2. 读取图片
    img = cv2.imread(input_img_name)

    # 容错处理
    if img is None:
        print(f"错误：找不到 {input_img_name}，请检查图片路径！")
    else:
        print(f"成功读取图片: {input_img_name}，正在处理...")


        result_he = HE.global_histogram_equalization(img)  # 普通 HE
        result_msrcr = Retinex.MSRCR(img)

        # 1. 如果文件夹不存在，就自动创建一个
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"已创建新文件夹: {output_folder}")

        # 2. 定义保存的文件名 (使用 os.path.join 拼接路径，防止斜杠写错)
        # 这里的 result_he 是 BGR 格式，直接保存颜色是正确的
        path_he = os.path.join(output_folder, 'result_he.jpg')
        path_msr = os.path.join(output_folder, 'result_msrcr.jpg')

        # 3. 写入文件
        cv2.imwrite(path_he, result_he)
        cv2.imwrite(path_msr, result_msrcr)

        print(f"保存成功！图片已存入 '{output_folder}' 文件夹。")

        plt.show()