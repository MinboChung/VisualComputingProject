import nibabel as nib
import numpy as np
import pyvista as pv

# ---------------------------
# 1. Load the NIfTI Scalar and Tensor Data
# ---------------------------
nii_map = nib.load("/Users/zhang/PycharmProjects/Visual_Computing_Project/BeginwithPyvista/Dataset/brain-map.nii.gz")
data_map = nii_map.get_fdata()
print("Brain map shape:", data_map.shape)

# ---------------------------
# 2. Create PyVista ImageData for Scalar Field
# ---------------------------
dims = data_map.shape
spacing = (1.0, 1.0, 1.0)
origin = (0.0, 0.0, 0.0)

grid = pv.ImageData(dimensions=dims, spacing=spacing, origin=origin)
grid.point_data["values"] = data_map.flatten(order="F")

# ---------------------------
# 3. Extract the Full Model and Apply Clipping for CT Effect
# ---------------------------
initial_contours = grid.contour([np.percentile(data_map, 90)])  # Extract isosurface
current_height = dims[2] // 2  # Default slice height

clipped_model = initial_contours.clip(normal=[0, 0, 1], origin=[0, 0, current_height])

# ---------------------------
# 4. Set Up PyVista Plotter
# ---------------------------
plotter = pv.Plotter()
plotter.add_axes()

# Add the clipped model
model_actor = plotter.add_mesh(clipped_model, color='white', smooth_shading=True, name="clipped_model")


# ---------------------------
# 5. Update Clipping Plane Based on Slice Height
# ---------------------------
def update_clip(value):
    global current_height, model_actor
    current_height = value
    plotter.remove_actor("clipped_model", reset_camera=False)
    clipped_model = initial_contours.clip(normal=[0, 0, 1], origin=[0, 0, current_height])
    model_actor = plotter.add_mesh(clipped_model, color='white', smooth_shading=True, name="clipped_model")
    plotter.render()


plotter.add_slider_widget(
    callback=update_clip,
    rng=[0, dims[2]],
    value=current_height,
    title="CT Slice Height",
    style="modern",
    pointa=(0.025, 0.9),
    pointb=(0.25, 0.9)
)


# ---------------------------
# 6. Adjust Lighting
# ---------------------------
def update_lighting(value):
    plotter.lighting.intensity = value
    plotter.render()


plotter.add_slider_widget(
    callback=update_lighting,
    rng=[0.1, 2.0],
    value=1.0,
    title="Lighting Intensity",
    style="modern",
    pointa=(0.025, 0.1),
    pointb=(0.25, 0.1)
)

# ---------------------------
# 7. Toggle Lighting Mode
# ---------------------------
lighting_modes = ["default", "none", "three lights"]
current_mode_idx = 0


def toggle_lighting():
    global current_mode_idx
    current_mode_idx = (current_mode_idx + 1) % len(lighting_modes)
    mode = lighting_modes[current_mode_idx]

    if mode == "default":
        plotter.enable_lightkit()
    elif mode == "none":
        plotter.disable_lighting()
    elif mode == "three lights":
        plotter.enable_lighting()
        light = pv.Light(position=(1, 1, 1), intensity=1.5)
        plotter.add_light(light)

    print(f"Lighting mode switched to: {mode}")
    plotter.render()


plotter.add_checkbox_button_widget(toggle_lighting, position=(50, 10))

# ---------------------------
# 8. Show the Interactive Visualization
# ---------------------------
plotter.show()