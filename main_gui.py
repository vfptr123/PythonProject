import sys
import cv2
import numpy as np
import HE
import Retinex
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QCheckBox, QFileDialog, QMessageBox, 
                             QScrollArea, QFrame, QGraphicsDropShadowEffect)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage, QColor

class ImageEnhancementApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setObjectName("MainWindow")
        self.current_image = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('图像增强算法对比系统 - 毕业设计')
        self.resize(1300, 850)
        
        # =========================================================
        # macOS Big Sur / Glassmorphism 风格样式表
        # =========================================================
        self.setStyleSheet("""
            /* 1. 全局背景：多彩渐变 */
            QWidget#MainWindow {
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1, stop:0 #E0C3FC, stop:1 #8EC5FC);
            }
            
            QWidget {
                font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
                color: #333333;
            }

            /* 2. 毛玻璃卡片容器 */
            QFrame#OptionFrame, QWidget#ImageCard {
                background-color: rgba(255, 255, 255, 180); /* 70% 透明度白色 */
                border: 1px solid rgba(255, 255, 255, 240); /* 亮白色边框，增强立体感 */
                border-radius: 20px;
            }

            /* 3. 复选框 */
            QCheckBox {
                font-size: 15px;
                color: #2c3e50;
                spacing: 8px;
                background-color: transparent;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border-radius: 6px;
                border: 1px solid #bdc3c7;
                background-color: rgba(255, 255, 255, 150);
            }
            QCheckBox::indicator:checked {
                background-color: #007AFF;
                border-color: #007AFF;
            }

            /* 4. 主按钮 */
            QPushButton#PrimaryButton {
                background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 #007AFF, stop:1 #0062CC);
                color: white;
                border-radius: 12px;
                padding: 12px 24px;
                font-size: 16px;
                font-weight: 600;
                border: 1px solid rgba(255, 255, 255, 50);
            }
            QPushButton#PrimaryButton:hover {
                background-color: #0088FF;
            }

            /* 5. 次要按钮 */
            QPushButton#SecondaryButton {
                background-color: rgba(255, 255, 255, 150);
                color: #333;
                border: 1px solid rgba(255, 255, 255, 200);
                border-radius: 12px;
                padding: 10px 20px;
                font-size: 15px;
                font-weight: 600;
            }
            QPushButton#SecondaryButton:hover {
                background-color: rgba(255, 255, 255, 220);
                color: #007AFF;
            }

            /* 6. 滚动区透明 */
            QScrollArea {
                border: none;
                background-color: transparent;
            }
            QWidget#DisplayWidget {
                background-color: transparent;
            }

            /* 7. 标题文字 */
            QLabel#MainTitle {
                font-size: 32px; 
                font-weight: 800; 
                color: #FFFFFF;
                margin-bottom: 10px;
            }
            
            QLabel#CardTitle {
                font-size: 16px;
                font-weight: 600;
                color: #2c3e50;
            }

            /* 8. 图片边框 (相框效果) */
            QLabel#ImageLabel {
                background-color: #F0F0F0;
                border: 4px solid #FFFFFF; /* 白色相框边框 */
                border-radius: 8px;
            }
        """)

        # 主布局
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(40, 40, 40, 40)
        main_layout.setSpacing(20)

        # ---------------------------------------------------------
        # 1. 标题区
        # ---------------------------------------------------------
        title_label = QLabel("图像增强算法对比平台")
        title_label.setObjectName("MainTitle")
        title_label.setAlignment(Qt.AlignCenter)

        title_shadow = QGraphicsDropShadowEffect()
        title_shadow.setBlurRadius(10)
        title_shadow.setColor(QColor(0, 0, 0, 60))
        title_shadow.setOffset(0, 2)
        title_label.setGraphicsEffect(title_shadow)

        main_layout.addWidget(title_label)

        # ---------------------------------------------------------
        # 2. 顶部选项区
        # ---------------------------------------------------------
        options_frame = QFrame()
        options_frame.setObjectName("OptionFrame")
        self.add_shadow(options_frame)

        options_layout = QHBoxLayout(options_frame)
        options_layout.setContentsMargins(40, 20, 40, 20)
        options_layout.setSpacing(40)
        
        self.cb_he = QCheckBox('直方图均衡 (HE)')
        self.cb_retinex = QCheckBox('Retinex')
        self.cb_new = QCheckBox('新算法 (预留)')
        self.cb_all = QCheckBox('全选')
        
        self.cb_he.setChecked(True)
        self.cb_retinex.setChecked(True)

        options_layout.addWidget(self.cb_he)
        options_layout.addWidget(self.cb_retinex)
        options_layout.addWidget(self.cb_new)
        options_layout.addStretch()
        
        self.cb_all.setStyleSheet("font-weight: bold; color: #007AFF;")
        options_layout.addWidget(self.cb_all)

        self.cb_all.stateChanged.connect(self.on_select_all)
        self.cb_he.stateChanged.connect(self.on_algo_change)
        self.cb_retinex.stateChanged.connect(self.on_algo_change)
        self.cb_new.stateChanged.connect(self.on_algo_change)

        main_layout.addWidget(options_frame)

        # ---------------------------------------------------------
        # 3. 上传按钮区
        # ---------------------------------------------------------
        btn_layout = QHBoxLayout()
        self.btn_upload = QPushButton('📂  点击上传图片')
        self.btn_upload.setObjectName("PrimaryButton")
        self.btn_upload.setCursor(Qt.PointingHandCursor)
        self.btn_upload.setFixedWidth(260)
        self.btn_upload.clicked.connect(self.open_image)

        btn_shadow = QGraphicsDropShadowEffect()
        btn_shadow.setBlurRadius(20)
        btn_shadow.setColor(QColor(0, 122, 255, 80))
        btn_shadow.setOffset(0, 8)
        self.btn_upload.setGraphicsEffect(btn_shadow)

        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_upload)
        btn_layout.addStretch()
        main_layout.addLayout(btn_layout)

        # ---------------------------------------------------------
        # 4. 核心展示区 (水平滚动)
        # ---------------------------------------------------------
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        self.display_widget = QWidget()
        self.display_widget.setObjectName("DisplayWidget")

        self.display_layout = QHBoxLayout(self.display_widget)
        self.display_layout.setAlignment(Qt.AlignLeft | Qt.AlignVCenter) # 垂直居中
        self.display_layout.setSpacing(40)
        self.display_layout.setContentsMargins(20, 20, 20, 20)

        scroll_area.setWidget(self.display_widget)
        main_layout.addWidget(scroll_area)

        # ---------------------------------------------------------
        # 5. 底部操作区
        # ---------------------------------------------------------
        bottom_layout = QHBoxLayout()
        self.btn_reupload = QPushButton('🔄  更换图片')
        self.btn_reupload.setObjectName("SecondaryButton")
        self.btn_reupload.setCursor(Qt.PointingHandCursor)
        self.btn_reupload.clicked.connect(self.open_image)

        bottom_layout.addStretch()
        bottom_layout.addWidget(self.btn_reupload)
        bottom_layout.addStretch()

        main_layout.addLayout(bottom_layout)

        self.setLayout(main_layout)

    def add_shadow(self, widget):
        """给控件添加柔和的阴影"""
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(25)
        shadow.setColor(QColor(0, 0, 0, 30))
        shadow.setOffset(0, 5)
        widget.setGraphicsEffect(shadow)

    def create_image_card(self, title, img_data):
        """创建毛玻璃风格的图片卡片，并绑定图片数据"""
        card = QWidget()
        card.setObjectName("ImageCard")
        self.add_shadow(card)

        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(10)
        
        # 图片 Label
        lbl_img = QLabel()
        lbl_img.setObjectName("ImageLabel")
        lbl_img.setAlignment(Qt.AlignCenter)

        # 绑定原始 OpenCV 数据到控件，方便后续 Resize 使用
        lbl_img.cv_image = img_data

        # 初始显示
        self.update_label_pixmap(lbl_img)

        # 标题 Label
        lbl_text = QLabel(title)
        lbl_text.setObjectName("CardTitle")
        lbl_text.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(lbl_img)
        layout.addWidget(lbl_text)
        card.setLayout(layout)

        return card, lbl_img

    def update_label_pixmap(self, label):
        """根据当前窗口高度，自适应调整图片大小"""
        if not hasattr(label, 'cv_image') or label.cv_image is None:
            return

        img = label.cv_image
        h, w, ch = img.shape

        # 核心逻辑：高度设置为窗口高度的 45%
        target_h = int(self.height() * 0.45)
        if target_h < 200: target_h = 200 # 最小高度限制

        scale = target_h / h
        target_w = int(w * scale)

        img_resized = cv2.resize(img, (target_w, target_h))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        bytes_per_line = 3 * target_w
        q_img = QImage(img_rgb.data, target_w, target_h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        label.setPixmap(pixmap)

    def resizeEvent(self, event):
        """窗口大小改变时，触发图片重绘"""
        # 遍历所有卡片，更新图片大小
        for i in range(self.display_layout.count()):
            item = self.display_layout.itemAt(i)
            if item:
                widget = item.widget()
                if widget:
                    # 查找卡片内的 ImageLabel
                    lbl = widget.findChild(QLabel, "ImageLabel")
                    if lbl:
                        self.update_label_pixmap(lbl)

        super().resizeEvent(event)

    # --- 逻辑处理函数 ---

    def on_select_all(self, state):
        is_checked = (state == Qt.Checked)
        self.cb_he.blockSignals(True)
        self.cb_retinex.blockSignals(True)
        self.cb_new.blockSignals(True)

        self.cb_he.setChecked(is_checked)
        self.cb_retinex.setChecked(is_checked)
        self.cb_new.setChecked(is_checked)

        self.cb_he.blockSignals(False)
        self.cb_retinex.blockSignals(False)
        self.cb_new.blockSignals(False)

        self.update_display()

    def on_algo_change(self):
        self.update_display()

    def open_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, '选择图片', '', 'Image files (*.jpg *.png *.jpeg *.bmp)')
        if fname:
            img = cv2.imread(fname)
            if img is None:
                QMessageBox.warning(self, "错误", "无法读取图片，请检查路径！")
                return

            self.current_image = img
            self.update_display()

    def update_display(self):
        """核心刷新逻辑"""
        if self.current_image is None:
            return

        # 清空布局
        while self.display_layout.count():
            item = self.display_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        # 1. 添加原图
        card, _ = self.create_image_card("原始图片", self.current_image)
        self.display_layout.addWidget(card)

        # 2. 根据勾选状态添加结果图
        if self.cb_he.isChecked():
            res_img = self.run_he(self.current_image.copy())
            card, _ = self.create_image_card("直方图均衡 (HE)", res_img)
            self.display_layout.addWidget(card)

        if self.cb_retinex.isChecked():
            res_img = self.run_retinex(self.current_image.copy())
            card, _ = self.create_image_card("Retinex 算法", res_img)
            self.display_layout.addWidget(card)

        if self.cb_new.isChecked():
            res_img = self.run_new_algo(self.current_image.copy())
            card, _ = self.create_image_card("新算法 (预留)", res_img)
            self.display_layout.addWidget(card)

    # =========================================================
    # 算法槽函数框架
    # =========================================================

    def run_he(self, img):
        # TODO: 接入真实算法
        #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return HE.global_histogram_equalization(img)

    def run_retinex(self, img):
        # TODO: 接入真实算法
        #return cv2.bitwise_not(img)
        return Retinex.MSRCR(img)

    def run_new_algo(self, img):
        # TODO: 接入真实算法
        return
        #return cv2.GaussianBlur(img, (25, 25), 0)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageEnhancementApp()
    ex.show()
    sys.exit(app.exec_())