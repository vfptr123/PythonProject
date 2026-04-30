"""
Zero-DCE 训练记录器与学术论文绘图
==================================
职责：
1. 逐 epoch 记录各项 loss（JSON 持久化，断点可续）
2. 训练结束后自动生成学术论文风格的收敛曲线图

用法（在 lowlight_train.py 中调用）：
    from training_logger import TrainingLogger
    logger = TrainingLogger(save_dir='logs/')
    # 训练循环中：
    logger.log_iteration(loss_dict)
    # 每个 epoch 结束：
    logger.end_epoch(epoch)
    # 训练结束：
    logger.plot()
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无 GUI 环境也能画图
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ==========================================
# 学术论文绘图风格配置
# ==========================================

def _setup_academic_style():
    """配置学术论文级别的 matplotlib 全局样式
    参考 IEEE / CVPR / ECCV 论文图表规范"""
    plt.rcParams.update({
        # 字体：优先 Times New Roman（论文标配）
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'SimSun'],
        'font.size': 11,

        # 数学字体
        'mathtext.fontset': 'stix',

        # 坐标轴
        'axes.linewidth': 0.8,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'axes.titleweight': 'bold',
        'axes.unicode_minus': False,

        # 刻度
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.minor.visible': True,
        'ytick.minor.visible': True,
        'xtick.minor.width': 0.4,
        'ytick.minor.width': 0.4,

        # 图例
        'legend.fontsize': 9,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '#cccccc',
        'legend.fancybox': False,

        # 网格
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'grid.linestyle': '--',

        # 线条
        'lines.linewidth': 1.5,
        'lines.markersize': 4,

        # 图像输出
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'figure.dpi': 150,
    })


# 学术论文配色（冷静克制，区分度高）
ACADEMIC_COLORS = {
    'total':   '#2c3e50',  # 深灰蓝 — 总 loss 主线
    'L_spa':   '#2980b9',  # 钴蓝
    'L_color': '#8e44ad',  # 紫罗兰
    'L_exp':   '#16a085',  # 青绿
    'L_TV':    '#d35400',  # 赭橙
}

LOSS_LABELS = {
    'total':   r'$\mathcal{L}_{total}$',
    'L_spa':   r'$\mathcal{L}_{spa}$',
    'L_color': r'$\mathcal{L}_{color}$',
    'L_exp':   r'$\mathcal{L}_{exp}$',
    'L_TV':    r'$\mathcal{L}_{TV}$',
}

LOSS_TITLES = {
    'total':   'Total Loss',
    'L_spa':   'Spatial Consistency Loss',
    'L_color': 'Color Constancy Loss',
    'L_exp':   'Exposure Control Loss',
    'L_TV':    'Total Variation Loss',
}


# ==========================================
# TrainingLogger 核心类
# ==========================================

class TrainingLogger:
    """训练过程记录器，支持断点续训"""

    def __init__(self, save_dir='logs/'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.log_path = os.path.join(save_dir, 'training_log.json')
        self.history = []          # 每个 epoch 的平均 loss
        self._epoch_buffer = []    # 当前 epoch 内各 iteration 的 loss

        # 尝试从已有日志恢复
        if os.path.exists(self.log_path):
            with open(self.log_path, 'r', encoding='utf-8') as f:
                self.history = json.load(f)
            print(f"[Logger] 已恢复 {len(self.history)} 个 epoch 的历史记录")

    def log_iteration(self, loss_dict: dict):
        """记录单次 iteration 的 loss
        
        Args:
            loss_dict: {'L_spa': float, 'L_color': float, 'L_exp': float, 'L_TV': float, 'total': float}
        """
        self._epoch_buffer.append({k: float(v) for k, v in loss_dict.items()})

    def end_epoch(self, epoch: int):
        """结束当前 epoch，计算平均 loss 并持久化"""
        if not self._epoch_buffer:
            return

        # 计算当前 epoch 所有 iteration 的平均值
        keys = self._epoch_buffer[0].keys()
        avg = {}
        for k in keys:
            vals = [d[k] for d in self._epoch_buffer]
            avg[k] = sum(vals) / len(vals)

        record = {'epoch': epoch, **avg}
        self.history.append(record)
        self._epoch_buffer.clear()

        # 持久化到 JSON
        with open(self.log_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)

        # 打印摘要
        loss_str = ' | '.join(f'{k}: {v:.4f}' for k, v in avg.items())
        print(f"[Epoch {epoch:>3d}] {loss_str}")

    def plot(self, output_path=None):
        """生成学术论文风格的训练曲线图
        
        生成两张图：
        1. 总 loss 收敛曲线（单图，适合正文）
        2. 各分量 loss 详细对比（2×2 子图，适合附录或补充材料）
        """
        if not self.history:
            print("[Logger] 无训练记录，跳过绘图")
            return

        _setup_academic_style()

        if output_path is None:
            output_path = self.save_dir

        self._plot_total_loss(output_path)
        self._plot_component_losses(output_path)
        self._plot_all_in_one(output_path)

    def _plot_total_loss(self, output_dir):
        """图1: 总 loss 收敛曲线（单图，论文正文用）"""
        epochs = [h['epoch'] for h in self.history]
        total = [h['total'] for h in self.history]

        fig, ax = plt.subplots(figsize=(5.5, 3.8))

        # 主线
        ax.plot(epochs, total,
                color=ACADEMIC_COLORS['total'],
                linewidth=1.8,
                zorder=3)

        # 淡色填充区域增强视觉
        ax.fill_between(epochs, total, alpha=0.06,
                        color=ACADEMIC_COLORS['total'])

        # 标注起止点
        ax.scatter([epochs[0], epochs[-1]], [total[0], total[-1]],
                   color=ACADEMIC_COLORS['total'], s=25, zorder=4)
        ax.annotate(f'{total[0]:.3f}',
                    xy=(epochs[0], total[0]),
                    xytext=(8, 8), textcoords='offset points',
                    fontsize=8, color='#555555')
        ax.annotate(f'{total[-1]:.3f}',
                    xy=(epochs[-1], total[-1]),
                    xytext=(-35, 8), textcoords='offset points',
                    fontsize=8, color='#555555')

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.set_title('Training Convergence Curve')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True)

        path = os.path.join(output_dir, 'loss_total.png')
        fig.savefig(path, facecolor='white')
        plt.close(fig)
        print(f"[Logger] 总 loss 曲线 → {path}")

    def _plot_component_losses(self, output_dir):
        """图2: 四分量 loss 子图（2×2 布局，论文详细分析用）"""
        epochs = [h['epoch'] for h in self.history]
        components = ['L_spa', 'L_color', 'L_exp', 'L_TV']

        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        fig.suptitle('Training Loss Components', fontsize=14, fontweight='bold', y=0.98)

        for ax, key in zip(axes.flat, components):
            values = [h.get(key, 0) for h in self.history]
            color = ACADEMIC_COLORS[key]

            ax.plot(epochs, values, color=color, linewidth=1.5)
            ax.fill_between(epochs, values, alpha=0.06, color=color)
            ax.set_title(LOSS_TITLES[key], fontsize=11)
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel('Loss', fontsize=10)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(True)

            # 标注最终值
            ax.annotate(f'{values[-1]:.4f}',
                        xy=(epochs[-1], values[-1]),
                        xytext=(-40, 8), textcoords='offset points',
                        fontsize=8, color=color, fontweight='bold')

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        path = os.path.join(output_dir, 'loss_components.png')
        fig.savefig(path, facecolor='white')
        plt.close(fig)
        print(f"[Logger] 分量 loss 子图 → {path}")

    def _plot_all_in_one(self, output_dir):
        """图3: 所有 loss 叠加在一张图上（归一化对比，论文对比分析用）"""
        epochs = [h['epoch'] for h in self.history]
        components = ['L_spa', 'L_color', 'L_exp', 'L_TV']

        fig, ax = plt.subplots(figsize=(6, 4))

        for key in components:
            values = [h.get(key, 0) for h in self.history]
            color = ACADEMIC_COLORS[key]
            label = LOSS_LABELS[key]
            ax.plot(epochs, values, color=color, linewidth=1.3,
                    label=label, alpha=0.9)

        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss Value')
        ax.set_title('Multi-component Loss Curves')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc='upper right', ncol=2)
        ax.grid(True)

        path = os.path.join(output_dir, 'loss_all_in_one.png')
        fig.savefig(path, facecolor='white')
        plt.close(fig)
        print(f"[Logger] 汇总对比曲线 → {path}")


# ==========================================
# 独立运行：从已有 JSON 日志重新绘图
# ==========================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='从已有训练日志重新生成论文图表')
    parser.add_argument('--log_dir', type=str, default='logs/',
                        help='训练日志目录（含 training_log.json）')
    args = parser.parse_args()

    logger = TrainingLogger(save_dir=args.log_dir)
    if logger.history:
        logger.plot()
    else:
        print("未找到训练日志，请先运行训练")
