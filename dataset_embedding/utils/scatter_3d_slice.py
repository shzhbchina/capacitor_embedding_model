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
    # --- 1. 三维散点图 ---
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df[x_col], df[y_col], df[z_col], c='b', alpha=0.7)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    plt.title(f'3D Scatter: {x_col} - {y_col} - {z_col}')
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