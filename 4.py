import vtk
import os
import nibabel as nib


def read_file(file_path):
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(file_path)
    reader.Update()
    return reader

# 读取 NIfTI 文件
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path_fa = os.path.join(script_dir, "Dataset", "brain-map.nii.gz")
file_path_tensor = os.path.join(script_dir, "Dataset", "brain-tensors.nii.gz")
file_path_vector = os.path.join(script_dir, "Dataset", "brain-vectors.nii.gz")

# 检查文件是否存在
for file_path in [file_path_fa, file_path_tensor, file_path_vector]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Error: File not found at {file_path}")
    
img_map = nib.load(file_path_fa)
img_vector = nib.load(file_path_tensor)
img_tensor = nib.load(file_path_vector)

# 获取数据数组
data_map = img_map.get_fdata()
data_vector = img_vector.get_fdata()
data_tensor = img_tensor.get_fdata()

# 查看数据的维度
print("Brain Map Shape:", data_map.shape)
print("Brain Vector Shape:", data_vector.shape)
print("Brain Tensor Shape:", data_tensor.shape)
