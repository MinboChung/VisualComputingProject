import nibabel as nib
import numpy as np

# 1. Load NIfTI files
# ----------------------------
# Load three different NIfTI datasets:
# - "brain-map.nii.gz": Scalar data
# - "brain-tensors.nii.gz": Tensor data (e.g., for DTI, often stored as 6 or 9 components per voxel)
# - "brain-vectors.nii.gz": Vector data (e.g., principal direction vectors)
nii_map = nib.load("/Users/zhang/PycharmProjects/Visual_Computing_Project/BeginwithPyvista/Dataset/brain-map.nii.gz")
nii_tensors = nib.load("/Users/zhang/PycharmProjects/Visual_Computing_Project/BeginwithPyvista/Dataset/brain-tensors.nii.gz")
nii_vectors = nib.load("/Users/zhang/PycharmProjects/Visual_Computing_Project/BeginwithPyvista/Dataset/brain-vectors.nii.gz")

# 2. Convert the loaded NIfTI images into NumPy arrays
# --------------------------------------------------------
# data_map: Expected shape is (X, Y, Z) for scalar data
# data_tensors: Expected shape is (X, Y, Z, 6/9) for tensor data (common in DTI)
# data_vectors: Expected shape is (X, Y, Z, 3) for vector data (e.g., principal vectors)
data_map = nii_map.get_fdata()
data_tensors = nii_tensors.get_fdata()
data_vectors = nii_vectors.get_fdata()

# 3. Print out the shapes of the arrays to verify dimensions
print("Brain map shape:", data_map.shape)
print("Brain tensors shape:", data_tensors.shape)
print("Brain vectors shape:", data_vectors.shape)

import nibabel as nib
import numpy as np
import pyvista as pv

# Load the NIfTI scalar data (brain map)
data_map = nii_map.get_fdata()

# Get the dimensions of the data (X, Y, Z)
dims = data_map.shape
spacing = (1.0, 1.0, 1.0)   # Set according to the actual voxel dimensions
origin = (0.0, 0.0, 0.0)    # Set the origin as needed

# Create a regular grid using PyVista's ImageData
grid = pv.ImageData(dimensions=dims, spacing=spacing, origin=origin)
# VTK requires the data to be flattened in Fortran (column-major) order
grid.point_data["values"] = data_map.flatten(order="F")

# Set an iso-value for isosurface extraction.
# Here, we use the 90th percentile of the intensity distribution.
iso_value = np.percentile(data_map, 90)
# Extract the isosurface using the specified iso-value
contours = grid.contour([iso_value])

# Visualize the isosurface
plotter = pv.Plotter()
plotter.add_mesh(contours, color='white')
plotter.add_axes()
plotter.show()
