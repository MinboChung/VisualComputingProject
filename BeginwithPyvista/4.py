import os
import nibabel as nib
import numpy as np
import pyvista as pv
import scipy.ndimage
from joblib import Parallel, delayed
import panel as pn
from pyvista import examples
from pyvista.plotting import system_supports_plotting

# ---------------------------
# 1. Load the NIfTI Scalar Data
# ---------------------------
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
smoothed_model = Parallel(n_jobs=-1)(delayed(initial_contours.smooth)(n_iter=2000) for _ in range(1))[0]
smoothed_model.compute_normals(cell_normals=True, point_normals=True, auto_orient_normals=True)

current_height = scaled_dims[2] // 2  # Default slice height
clipped_model = smoothed_model.clip(normal=[0, 0, 1], origin=[0, 0, current_height])

# ---------------------------
# 4. Set Up PyVista Panel Plotter
# ---------------------------
plotter = pv.Plotter()
plotter.add_axes()

# Ensure VTK rendering is available
if not system_supports_plotting():
    raise RuntimeError("VTK rendering is not supported on this system.")

# Add default light source
light = pv.Light(position=(1, 1, 1), intensity=0.8, light_type="headlight")
plotter.add_light(light)
model_actor = plotter.add_mesh(clipped_model, color='gray', smooth_shading=True, specular=0.2, name="clipped_model")

# ---------------------------
# 5. Interactive Controls with Panel
# ---------------------------
light_slider = pn.widgets.FloatSlider(name='Light Intensity', start=0.1, end=2.0, value=0.8, step=0.1)
color_selector = pn.widgets.Select(name='Model Color', options=['gray', 'blue', 'red', 'green', 'white'])
ct_toggle = pn.widgets.Toggle(name='CT Mode')
slice_slider = pn.widgets.IntSlider(name='CT Slice Height', start=0, end=scaled_dims[2], value=current_height)

# ---------------------------
# 6. Define Callback Functions
# ---------------------------
def update_light(event):
    light.intensity = event.new
    plotter.render()

def update_color(event):
    plotter.remove_actor("clipped_model", reset_camera=False)
    new_color = event.new
    plotter.add_mesh(clipped_model, color=new_color, smooth_shading=True, name="clipped_model")
    plotter.render()

def toggle_ct(event):
    plotter.remove_actor("ct_slice", reset_camera=False)
    if event.new:
        ct_slice = resampled_grid.slice(normal=[0, 0, 1], origin=[0, 0, current_height])
        if ct_slice.n_points > 0:
            plotter.add_mesh(ct_slice, cmap="gray", opacity=1.0, name="ct_slice")
    plotter.render()

def update_slice(event):
    global current_height
    current_height = event.new
    plotter.remove_actor("clipped_model", reset_camera=False)
    clipped_model = smoothed_model.clip(normal=[0, 0, 1], origin=[0, 0, current_height])
    plotter.add_mesh(clipped_model, color='gray', smooth_shading=True, name="clipped_model")
    plotter.render()

# Link Widgets to Functions
light_slider.param.watch(update_light, 'value')
color_selector.param.watch(update_color, 'value')
ct_toggle.param.watch(toggle_ct, 'value')
slice_slider.param.watch(update_slice, 'value')

# ---------------------------
# 7. Deploy Web Application
# ---------------------------
layout = pn.Column(
    light_slider,
    color_selector,
    ct_toggle,
    slice_slider,
    pn.pane.VTK(plotter.ren_win, sizing_mode='stretch_width')
)

pn.serve(layout, port=5006)

