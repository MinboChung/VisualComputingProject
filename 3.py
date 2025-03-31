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

# **边缘检测（Edge Detection）**
# 计算网格的法线
normals = largest_component.compute_normals(cell_normals=False, inplace=False)

# 通过法线的变化来实现边缘检测（可以自定义阈值来判断边缘）
edge_threshold = 0.7  # 阈值，可以根据需求调整
edges = largest_component.extract_all_edges()

# **渲染**
plotter = pv.Plotter()

# 渲染曲率
# plotter.add_mesh(largest_component, scalars="Curvature", cmap="coolwarm", smooth_shading=True)

# 渲染边缘
plotter.add_mesh(edges, color="white", line_width=2)

plotter.set_background("black")
plotter.show()
