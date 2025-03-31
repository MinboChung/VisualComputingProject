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

# **提取等值面（Marching Cubes）**
iso_value = 0.5
contour = grid.contour([iso_value])  # 提取等值面，返回 PolyData

# **进行连通组件分析，去除悬浮颗粒物**
largest_component = contour.connectivity(largest=True)

# **计算法线（Phong 需要法线）**
largest_component.compute_normals(cell_normals=False, point_normals=True, inplace=True)

# **Phong 渲染**
plotter = pv.Plotter()
plotter.set_background("black")

plotter.add_mesh(
    largest_component,
    color="lightblue",      # 物体颜色
    smooth_shading=True,    # 开启平滑着色
    specular=1.0,           # 镜面反射强度
    specular_power=50.0,    # 镜面高光锐度（数值越高，高光越小越亮）
    ambient=0.2,            # 环境光
    diffuse=0.8             # 漫反射
)

plotter.show()
