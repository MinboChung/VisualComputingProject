import os
import nibabel as nib
import numpy as np
import pyvista as pv
import scipy.ndimage
from joblib import Parallel, delayed

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "Dataset", "brain-map.nii.gz")
nii_map = nib.load(file_path)
data_map = nii_map.get_fdata()
print("Brain map shape:", data_map.shape)

# ---------------------------
# 2. Resample Data for Higher Resolution
# ---------------------------
scaled_data = scipy.ndimage.zoom(data_map, (2, 2, 2), order=1, mode='nearest')  # Bilinear interpolation
scaled_dims = scaled_data.shape

resampled_grid = pv.ImageData(dimensions=scaled_dims, spacing=(0.5, 0.5, 0.5))
resampled_grid.point_data["values"] = scaled_data.flatten(order="F")

# ---------------------------
# 3. Extract Smoothed Model
# ---------------------------
initial_contours = resampled_grid.contour([np.percentile(scaled_data, 90)])
# Parallelize smoothing
smoothed_model = Parallel(n_jobs=-1)(delayed(initial_contours.smooth)(n_iter=2000) for _ in range(1))[0]

# Remove small fragments and keep only the largest connected part
# Use joblib for parallel connectivity filtering
cleaned_model = Parallel(n_jobs=-1)(delayed(smoothed_model.connectivity)(extraction_mode='largest') for _ in range(1))[0]
# Apply parallel smoothing again
cleaned_model = Parallel(n_jobs=-1)(delayed(cleaned_model.smooth)(n_iter=1000, relaxation_factor=0.9) for _ in range(1))[0]
smoothed_model.compute_normals(cell_normals=True, point_normals=True, auto_orient_normals=True)

current_height = scaled_dims[2] // 2  # Default slice height
clipped_model = smoothed_model.clip(normal=[0, 0, 1], origin=[0, 0, current_height])

# ---------------------------
# 4. Set Up PyVista Plotter with Improved Lighting
# ---------------------------
plotter = pv.Plotter()
plotter.add_axes()

# Add a stronger light source
light = pv.Light(position=(1, 1, 1), intensity=0.8, light_type="headlight")
plotter.add_light(light)

# Add ambient light to balance lighting
# Add soft scene lights to simulate ambient lighting
scene_light1 = pv.Light(position=(-1, -1, -1), intensity=0.3, light_type='scenelight')
scene_light2 = pv.Light(position=(1, -1, 1), intensity=0.3, light_type='scenelight')
plotter.add_light(scene_light1)
plotter.add_light(scene_light2)


# Add the clipped model with better shading
model_actor = plotter.add_mesh(clipped_model, color='gray', smooth_shading=True, specular=0.2, name="clipped_model")

# ---------------------------
# 5. Update Clipping Plane Based on Slice Height
# ---------------------------
def update_clip(value):
    global current_height, model_actor
    current_height = value
    plotter.remove_actor("clipped_model", reset_camera=False)
    clipped_model = smoothed_model.clip(normal=[0, 0, 1], origin=[0, 0, current_height])
    model_actor = plotter.add_mesh(clipped_model, color='gray', smooth_shading=True, name="clipped_model")
    plotter.render()

plotter.add_slider_widget(
    callback=update_clip,
    rng=[0, scaled_dims[2]],
    value=current_height,
    title="CT Slice Height",
    style="modern",
    pointa=(0.025, 0.9),
    pointb=(0.25, 0.9)
)

# ---------------------------
# 6. Show the Interactive Visualization
# ---------------------------
plotter.show()
