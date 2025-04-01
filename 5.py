import pyvista as pv
import nibabel as nib
import numpy as np
import os

def read_nifti(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError("Error: Missing required NIfTI files.")
    nii = nib.load(file_path)
    data = nii.get_fdata()
    return data

# Read files
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path_map = os.path.join(script_dir, "Dataset", "brain-map.nii.gz")
file_path_tensor = os.path.join(script_dir, "Dataset", "brain-tensors.nii.gz")  # 读取张量数据
file_path_vector = os.path.join(script_dir, "Dataset", "brain-vectors.nii.gz")  # 或者使用向量数据

map_data = read_nifti(file_path_map)
tensor_data = read_nifti(file_path_tensor)
vector_data = read_nifti(file_path_vector)

def compute_FA_MD(tensor_data):
    """
    计算 FA（Fractional Anisotropy）、MD（Mean Diffusivity）和主方向
    tensor_data: (X, Y, Z, 6) -> 扩散张量数据
    
    # 定义： FA = 各向异性分数, MD = 平均扩散率, AD = 轴向扩散率, CSF = 脑脊液 ,  GM = 灰质,  RD = 径向扩散率, SNR = 信噪比, WM = 白质, λ = 特征值；张量中轴的长度

    """
    X, Y, Z, _ = tensor_data.shape
    
    # 提取 6 个分量
    Dxx, Dyy, Dzz, Dxy, Dxz, Dyz = np.moveaxis(tensor_data, -1, 0)

    # 构造 3×3 扩散张量 (X, Y, Z, 3, 3)
    D = np.zeros((X, Y, Z, 3, 3))
    D[..., 0, 0] = Dxx
    D[..., 1, 1] = Dyy
    D[..., 2, 2] = Dzz
    D[..., 0, 1] = D[..., 1, 0] = Dxy
    D[..., 0, 2] = D[..., 2, 0] = Dxz
    D[..., 1, 2] = D[..., 2, 1] = Dyz

    # 处理 NaN 和 Inf
    D[np.isnan(D)] = 0
    D[np.isinf(D)] = 0


    symmetry_check = np.allclose(D, np.swapaxes(D, -2, -1))
    print("D 是否对称:", symmetry_check)

    # 计算特征值和特征向量
    eigvals, eigvecs = np.linalg.eigh(D)  # 计算 3 个特征值和特征向量
    print("最小特征值:", np.min(eigvals))
    print("最大特征值:", np.max(eigvals))


    # 按大小排序 λ1 >= λ2 >= λ3
    eigvals = np.sort(eigvals, axis=-1)[..., ::-1]  
    λ1, λ2, λ3 = eigvals[..., 0], eigvals[..., 1], eigvals[..., 2]

    # 计算 MD（Mean Diffusivity）
    MD = (λ1 + λ2 + λ3) / 3
    print("MD 最小值:", np.min(MD))
    print("MD 最大值:", np.max(MD))

    # 计算 FA（Fractional Anisotropy）
    numerator = np.sqrt(1.5 * ((λ1 - MD) ** 2 + (λ2 - MD) ** 2 + (λ3 - MD) ** 2))
    denominator = np.sqrt(λ1**2 + λ2**2 + λ3**2)
    FA = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator != 0)

    print("FA 最小值:", np.min(FA))
    print("FA 最大值:", np.max(FA))

    return FA, MD, eigvecs

print(type(tensor_data))
print(tensor_data.shape)
print("Tensor data min:", np.min(tensor_data))
print("Tensor data max:", np.max(tensor_data))
print("Tensor data mean:", np.mean(tensor_data))
# 创建一个布尔掩码，标记既不是 NaN 也不是 0 的数据
not_nan_or_zero_data = tensor_data[~np.isnan(tensor_data) & (tensor_data != 0)]

# 打印出来
print("既不是 NaN 也不是 0 的数据：", not_nan_or_zero_data)
# FA,MD,eigvecs = compute_FA_MD(tensor_data)