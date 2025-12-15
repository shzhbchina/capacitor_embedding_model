import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def scatter_3d_slice(df, x_col, y_col, z_col,
                     highlight_col=None, highlight_value=None):
    """
    绘制三维散点图，xyz轴名称均可自定义；
    可按某列值±10%范围筛选并画二维投影。

    参数:
    - df: pandas.DataFrame
    - x_col, y_col, z_col: str，三轴所用的列名
    - highlight_col: str, 用于筛选的列名（可选，xyz轴中的某一个）
    - highlight_value: float, 筛选目标值（可选）
    """
    # # --- 1. 三维散点图 ---
    # fig = plt.figure(figsize=(9, 7))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(df[x_col], df[y_col], df[z_col], c='b', alpha=0.7)
    # ax.set_xlabel(x_col)
    # ax.set_ylabel(y_col)
    # ax.set_zlabel(z_col)
    # plt.title(f'3D Scatter: {x_col} - {y_col} - {z_col}')
    # plt.show()


    import numpy as np
    import matplotlib.ticker as ticker

    # --- 0. 模拟数据 (请替换为您真实的 df 数据) ---
    # 假设数据量较大
    # import pandas as pd
    # df = pd.DataFrame({
    #     x_col: np.random.uniform(10, 800, 1000),
    #     y_col: 10**np.random.uniform(0, 6, 1000), # 1uF 到 10^6 uF
    #     z_col: 10**np.random.uniform(2, 6, 1000)  # Volume
    # })

    # --- 1. 全局字体设置 ---
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.unicode_minus'] = False

    # --- 2. 创建画布 ---
    fig = plt.figure(figsize=(10, 8), dpi=120)
    ax = fig.add_subplot(111, projection='3d')

    # --- 3. 数据 Log 变换 (手动) ---
    z_log_data = np.log10(df[z_col])
    y_log_data = np.log10(df[y_col])

    # --- 4. 绘制散点 (关键优化：变小、变透、变细) ---
    # s=35: 显著减小点的大小 (原80 -> 35)
    # alpha=0.6: 增加透明度 (原0.8 -> 0.6)
    # linewidth=0.3: 边框变细 (原0.5 -> 0.3)，防止黑边糊成一团
    # edgecolor='k': 保持黑色边框增加轮廓，也可以试试 'w' (白色) 看是否更清爽
    scatter = ax.scatter(df[x_col], y_log_data, z_log_data,
                         c=z_log_data, cmap='viridis',
                         s=35, edgecolor='k', linewidth=0.3, alpha=0.6)

    # --- 5. 坐标轴格式化 (Y轴和Z轴都应用) ---
    def log_tick_formatter(x, pos):
        # 格式化为 10^x
        return f"$10^{{{int(x)}}}$"

    # Z轴设置
    ax.zaxis.set_major_formatter(ticker.FuncFormatter(log_tick_formatter))
    ax.zaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # 【修复】Y轴设置 (启用 Log 格式化)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(log_tick_formatter))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # --- 6. 坐标轴标签 ---
    ax.set_xlabel(x_col, fontsize=12, labelpad=10)
    ax.set_ylabel(f"{y_col} (Log Scale)", fontsize=12, labelpad=10)
    ax.set_zlabel(f"{z_col} (Log Scale)", fontsize=12, labelpad=12)

    # 刻度优化
    ax.tick_params(axis='both', which='major', labelsize=10, width=1, length=4)

    # --- 7. 网格线弱化 (避免喧宾夺主) ---
    grid_style = {'linewidth': 0.8, 'linestyle': '--', 'alpha': 0.5, 'color': 'gray'}
    ax.xaxis._axinfo["grid"].update(grid_style)
    ax.yaxis._axinfo["grid"].update(grid_style)
    ax.zaxis._axinfo["grid"].update(grid_style)

    # --- 8. 背景去色 ---
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')

    # --- 9. 调整视角 (可选) ---
    # elev: 仰角, azim: 方位角
    # 稍微调高 elev 可以减少前后点的遮挡
    ax.view_init(elev=30, azim=-60)

    # --- 10. 标题与色条 ---
    plt.title(f'3D Scatter Distribution', fontsize=16, pad=20)

    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.6)
    cbar.set_label(f"Log({z_col})", fontsize=12, fontweight='bold')
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(log_tick_formatter))

    plt.tight_layout()
    plt.show()






    # --- 2. 可选：二维投影 ---
    if highlight_col is not None and highlight_value is not None:
        lower = highlight_value * 0.9
        upper = highlight_value * 1.1
        mask = (df[highlight_col] >= lower) & (df[highlight_col] <= upper)
        filtered = df[mask]
        print(f"共筛选到 {len(filtered)} 个样本，{highlight_col} 在 [{lower:.2f}, {upper:.2f}]之间")
        # 只选xyz三轴中，除了highlight_col之外的两个
        axis = [x_col, y_col, z_col]
        axis.remove(highlight_col)
        x, y = filtered[axis[0]], filtered[axis[1]]

        plt.figure(figsize=(7, 5))
        plt.scatter(x, y, c='r', alpha=0.7)
        plt.xlabel(axis[0])
        plt.ylabel(axis[1])
        plt.title(f'2D Scatter: {axis[0]} vs {axis[1]} (where {highlight_col}≈{highlight_value})')
        plt.show()
        return filtered
    else:
        return None


# ----------- 用法举例 -----------
# 假设你有一个DataFrame叫df，三列名分别为"Voltage", "Volume", "Capacitance"
# plot_capacitor_3d(df, "Voltage", "Volume", "Capacitance")
# 举例：只筛选体积在100附近的数据
# plot_capacitor_3d(df, "Voltage", "Volume", "Capacitance", highlight_col="Volume", highlight_value=100)