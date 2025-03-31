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
file_path_map = os.path.join(script_dir, "Dataset", "brain-map.nii.gz")
file_path_tensor = os.path.join(script_dir, "Dataset", "brain-tensors.nii.gz")  # 读取张量数据
file_path_vector = os.path.join(script_dir, "Dataset", "brain-vectors.nii.gz")  # 或者使用向量数据

if not os.path.exists(file_path_map) or not os.path.exists(file_path_tensor):
    raise FileNotFoundError("Error: Missing required NIfTI files.")

# 读取 FA（体积数据）和张量或向量数据
map_data = read_nifti(file_path_map)
tensor_data = read_nifti(file_path_tensor)
vector_data = read_nifti(file_path_vector)

vector_data = nib.load(file_path_vector).get_fdata()

# 查看向量数据的形状
print(f"Vector data shape: {vector_data.shape}")

# 检查数据的最后一个维度是否为3（表示 x, y, z 分量）
if vector_data.shape[-1] == 3:
    print("Vector data contains 3D vectors.")
    
    # 打印部分数据以检查
    print("Example vector at (0, 0, 0):", vector_data[0, 0, 0, :])
    print("Example vector at (1, 1, 1):", vector_data[1, 1, 1, :])

else:
    print("Warning: Vector data does not appear to contain 3D vectors.")


# 处理数据
map_data = np.transpose(map_data, (2, 1, 0))  # 调整轴顺序
tensor_data = np.transpose(tensor_data, (2, 1, 0, 3))  # 处理张量数据
vector_data = np.transpose(vector_data, (2, 1, 0, 3))  # 处理向量数据

# **Step 1: 提取等值面（Marching Cubes）**
grid = pv.wrap(map_data)
iso_value = 0.5  # 适合 FA 数据的等值面值
contour = grid.contour([iso_value])

# **Step 2: 使用向量或张量计算表面法向量和曲率**
# 使用向量的模长来计算法线方向（假设向量数据包含了 3D 法向量）
vector_magnitude = np.linalg.norm(vector_data, axis=-1)  # 计算向量模长
curvature = contour.curvature()  # 计算曲率，假设这是表面曲率

# 检查曲率范围
print("Curvature range:", curvature.min(), curvature.max())

# **Step 3: 将颜色映射与曲率/法向量结合**
# 归一化曲率
curvature = (curvature - curvature.min()) / (curvature.max() - curvature.min())  # 曲率归一化到 [0, 1]
print("Normalized Curvature range:", curvature.min(), curvature.max())

# 将曲率作为颜色数据映射
contour["Curvature"] = curvature

# **Step 4: 渲染 - 间接体积渲染**
# 使用曲率进行颜色渲染
plotter = pv.Plotter()
plotter.add_mesh(contour, scalars="Curvature", cmap="coolwarm", smooth_shading=True)
plotter.set_background("black")
plotter.show()
