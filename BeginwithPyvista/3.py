import nibabel as nib
import numpy as np
import pyvista as pv

# ---------------------------
# 1. Load the NIfTI Scalar and Tensor Data
# ---------------------------
nii_map = nib.load("/Users/zhang/PycharmProjects/Visual_Computing_Project/BeginwithPyvista/Dataset/brain-map.nii.gz")
data_map = nii_map.get_fdata()
print("Brain map shape:", data_map.shape)

nii_tensors = nib.load(
    "/Users/zhang/PycharmProjects/Visual_Computing_Project/BeginwithPyvista/Dataset/brain-tensors.nii.gz")
data_tensors = nii_tensors.get_fdata()
print("Brain tensors shape:", data_tensors.shape)

nii_vectors = nib.load(
    "/Users/zhang/PycharmProjects/Visual_Computing_Project/BeginwithPyvista/Dataset/brain-vectors.nii.gz")
data_vectors = nii_vectors.get_fdata()
print("Brain vectors shape:", data_vectors.shape)

# ---------------------------
# 2. Create PyVista ImageData for Scalar Field
# ---------------------------
dims = data_map.shape
spacing = (1.0, 1.0, 1.0)
origin = (0.0, 0.0, 0.0)

grid = pv.ImageData(dimensions=dims, spacing=spacing, origin=origin)
grid.point_data["values"] = data_map.flatten(order="F")

# ---------------------------
# 3. Extract Isosurface using Marching Cubes
# ---------------------------
initial_iso = float(np.percentile(data_map, 90))
print(f"Using initial iso_value: {initial_iso:.3f}")
initial_contours = grid.contour([initial_iso])

# ---------------------------
# 4. Set Up PyVista Plotter
# ---------------------------
plotter = pv.Plotter()
actor = plotter.add_mesh(initial_contours, color='white', smooth_shading=True, name="iso_surface")
actor.GetProperty().SetInterpolationToPhong()
plotter.add_silhouette(initial_contours, color='black', line_width=2)
plotter.add_axes()

# ---------------------------
# 5. Visualize Tensor Data (Ellipsoids for Anisotropy)
# ---------------------------
if data_tensors.shape[-1] == 6:  # Check if tensor data has 6 components (symmetric tensor)
    tensor_points = []
    tensor_matrices = []

    for x in range(0, dims[0], 5):  # Sample points to reduce density
        for y in range(0, dims[1], 5):
            for z in range(0, dims[2], 5):
                tensor = data_tensors[x, y, z, :]
                tensor_matrix = np.array([[tensor[0], tensor[1], tensor[2]],
                                          [tensor[1], tensor[3], tensor[4]],
                                          [tensor[2], tensor[4], tensor[5]]])
                tensor_points.append([x, y, z])
                tensor_matrices.append(tensor_matrix.flatten())

    tensors = pv.PolyData(np.array(tensor_points))
    tensors.point_data["tensors"] = np.array(tensor_matrices)

    glyph_filter = tensors.glyph(scale=2, geom=pv.Sphere())  # Ensure glyph() is used correctly
    plotter.add_mesh(glyph_filter, color="blue", opacity=0.5, name="tensor_field")

# ---------------------------
# 6. Visualize Vector Field (Arrows for Direction)
# ---------------------------
if data_vectors.shape[-1] == 3:  # Check if vector field has 3 components
    vector_points = []
    vector_dirs = []

    for x in range(0, dims[0], 5):
        for y in range(0, dims[1], 5):
            for z in range(0, dims[2], 5):
                vector = data_vectors[x, y, z, :]
                vector_points.append([x, y, z])
                vector_dirs.append(vector)

    vectors = pv.PolyData(np.array(vector_points))
    vectors.point_data["vectors"] = np.array(vector_dirs)

    glyph_filter = vectors.glyph(orient="vectors", scale=3, geom=pv.Arrow())  # Fix glyph usage
    plotter.add_mesh(glyph_filter, color="red", name="vector_field")


# ---------------------------
# 7. Define the Slider Callback Function for Interactive Iso-Value Update
# ---------------------------
def update_iso_value(value):
    iso_val = float(value)
    new_contours = grid.contour([iso_val])
    plotter.remove_actor("iso_surface", reset_camera=False)
    plotter.add_mesh(new_contours, color='white', smooth_shading=True, name="iso_surface")
    plotter.render()


# ---------------------------
# 8. Add Interactive Slider Widget
# ---------------------------
plotter.add_slider_widget(
    callback=update_iso_value,
    rng=[data_map.min(), data_map.max()],
    value=initial_iso,
    title="Iso-Value",
    style="modern",
    pointa=(0.025, 0.1),
    pointb=(0.25, 0.1)
)

# ---------------------------
# 9. Show the Interactive Visualization
# ---------------------------
plotter.show()
