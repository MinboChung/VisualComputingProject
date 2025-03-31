import pyvista as pv
import nibabel as nib
import numpy as np
import os

def read_nifti(file_path):
    nii = nib.load(file_path)
    data = nii.get_fdata()
    return data

# 读取 NIfTI 文件
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path_fa = os.path.join(script_dir, "Dataset", "brain-map.nii.gz")

if not os.path.exists(file_path_fa):
    raise FileNotFoundError(f"Error: File not found at {file_path_fa}")

# 读取 FA 数据
fa_data = read_nifti(file_path_fa)

# PyVista 需要 (x, y, z) 形状的数据，因此调整轴顺序
fa_data = np.transpose(fa_data, (2, 1, 0))

# 创建 PyVista 的 ImageData 对象
grid = pv.wrap(fa_data)

# **先提取等值面（Marching Cubes），转换为 PolyData**
iso_value = 0.5
contour = grid.contour([iso_value])  # 提取等值面，返回 PolyData

# **进行连通组件分析，去除悬浮颗粒物**
largest_component = contour.connectivity(largest=True)

# **计算曲率（Curvature）**
largest_component["Curvature"] = largest_component.curvature("mean")

# **计算法线**
normals = largest_component.compute_normals(cell_normals=False, inplace=False)

# **自定义 Toon Shading（卡通渲染）**
# 计算光照：假设光源方向 (0, 0, 1)
light_direction = np.array([0, 0, 1])
dot_product = np.dot(normals.point_normals, light_direction)

# 离散化光照强度，实现 Toon Shading
toon_shading = np.digitize(dot_product, bins=[-1, -0.5, 0, 0.5, 1]) - 1

# 定义 Toon 颜色映射（RGB）
cmap = np.array([
    [0, 0, 255],  # 蓝色
    [0, 255, 0],  # 绿色
    [255, 255, 0],  # 黄色
    [255, 0, 0]   # 红色
]) / 255.0  # 归一化到 [0,1] 之间

# 确保 toon_shading 的索引在 cmap 的范围内
toon_shading = np.clip(toon_shading, 0, len(cmap) - 1)

# 映射到 RGB 颜色
toon_colors = cmap[toon_shading]

# **渲染**
plotter = pv.Plotter()

# 渲染 Toon Shading，注意 scalars 需要是 RGB 格式
plotter.add_mesh(largest_component, scalars=toon_colors, rgb=True, smooth_shading=True)

plotter.set_background("black")
plotter.show()
