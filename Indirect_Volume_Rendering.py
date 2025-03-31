import pyvista as pv
import nibabel as nib
import numpy as np
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path_map = os.path.join(script_dir, "Dataset", "brain-map.nii.gz")

def read_nifti(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: File not found at {file_path}")
    nii = nib.load(file_path)
    data = nii.get_fdata()
    return data


map_data = read_nifti(file_path_map)

# PyVista 需要 (x, y, z) 形状的数据，因此调整轴顺序
map_data = np.transpose(map_data, (2, 1, 0))

# 创建 PyVista 的 ImageData 对象
grid = pv.wrap(map_data)

# **先提取等值面（Marching Cubes），转换为 PolyData**
iso_value = 0.3
contour = grid.contour([iso_value])  # 提取等值面，返回 PolyData

# **进行连通组件分析，去除悬浮颗粒物**
largest_component = contour.connectivity(extraction_mode='largest')


# **计算法线（Phong 需要法线）**
normals = largest_component.compute_normals(cell_normals=False, inplace=False)


# **边缘检测（Edge Detection）**
# 通过法线的变化来实现边缘检测（可以自定义阈值来判断边缘）
edge_threshold = 0.7  # 阈值，可以根据需求调整
edges = largest_component.extract_all_edges()


# # **自定义 Toon Shading（卡通渲染）**
# # 计算光照：假设光源方向 (0, 0, 1)
# light_direction = np.array([0, 0, 1])
# dot_product = np.dot(normals.point_normals, light_direction)

# # 离散化光照强度，实现 Toon Shading
# toon_shading = np.digitize(dot_product, bins=[-1, -0.5, 0, 0.5, 1]) - 1

# # 定义 Toon 颜色映射（RGB）
# cmap = np.array([
#     [0, 0, 255],  # 蓝色
#     [0, 255, 0],  # 绿色
#     [255, 255, 0],  # 黄色
#     [255, 0, 0]   # 红色
# ]) / 255.0  # 归一化到 [0,1] 之间

# # 确保 toon_shading 的索引在 cmap 的范围内
# toon_shading = np.clip(toon_shading, 0, len(cmap) - 1)
# # 映射到 RGB 颜色
# toon_colors = cmap[toon_shading]




# **渲染**
plotter = pv.Plotter()
plotter.set_background("black")

plotter.add_mesh(
    largest_component,
    # scalars=toon_colors, 
    # rgb=True,
    color="lightblue",      # 物体颜色
    smooth_shading=True,    # 开启平滑着色
    specular=1.0,           # 镜面反射强度
    specular_power=50.0,    # 镜面高光锐度（数值越高，高光越小越亮）
    ambient=0.2,            # 环境光
    diffuse=0.8         ,    # 漫反射
)

# plotter.add_mesh(edges, color="orange", line_width=2)
plotter.show()
