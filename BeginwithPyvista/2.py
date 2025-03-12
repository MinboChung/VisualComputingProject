import nibabel as nib
import numpy as np
import pyvista as pv
import os


# Create a default PyVista Plotter to inspect the default light settings
plotter = pv.Plotter()
light_types = [light.light_type for light in plotter.renderer.lights]


def load_scalar_volume(file_path, spacing=(1.0, 1.0, 1.0), origin=(0.0, 0.0, 0.0)):
    """
    Load a scalar volume from a NIfTI file and convert it to a PyVista ImageData object.

    Parameters
    ----------
    file_path : str
        Path to the NIfTI file (e.g., 'brain-map.nii.gz').
    spacing : tuple of float, optional
        The spacing (voxel size) in each dimension (x, y, z).
    origin : tuple of float, optional
        The (x, y, z) origin of the dataset in 3D space.

    Returns
    -------
    grid : pv.ImageData
        A PyVista ImageData object representing the loaded 3D volume.
    data : numpy.ndarray
        The raw NumPy array containing the volume data.
    """
    # Load the NIfTI file using nibabel
    nii = nib.load(file_path)
    data = nii.get_fdata()

    # Get the dimensions of the 3D data (X, Y, Z)
    dims = data.shape

    # Create a PyVista ImageData object with the given spacing and origin
    grid = pv.ImageData(dimensions=dims, spacing=spacing, origin=origin)

    # Flatten the NumPy array in Fortran order to match VTK's internal memory layout
    grid.point_data["values"] = data.flatten(order="F")
    return grid, data


def extract_iso_surface(grid, iso_value):
    """
    Extract an isosurface (3D model) from the given volume data using a specific iso-value.

    Parameters
    ----------
    grid : pv.ImageData
        The PyVista ImageData object containing the volume.
    iso_value : float
        The threshold value used for isosurface extraction.

    Returns
    -------
    pv.PolyData
        A PolyData mesh that represents the surface at the given iso_value.
    """
    return grid.contour([iso_value])


def visualize_3d_model(iso_surface):
    """
    Visualize the 3D model (isosurface) in an interactive PyVista scene,
    adding custom lights and shadows.

    Parameters
    ----------
    iso_surface : pv.PolyData
        The 3D surface mesh to be visualized.
    """
    # Create a new Plotter for rendering
    plotter = pv.Plotter()

    # Add the isosurface to the scene with desired styling
    plotter.add_mesh(
        iso_surface,
        color='white',
        show_edges=True,  # Show polygon edges
        edge_color='black',  # Color for the edges
        smooth_shading=True  # Enable smooth shading
    )

    # Create a headlight that always shines from the camera's perspective
    head_light = pv.Light(light_type='headlight')
    plotter.add_light(head_light)

    # Create a custom light source positioned far away with a red color
    custom_light = pv.Light(
        position=(10000, 10000, 10000),  # Light position in 3D space
        focal_point=(-10, -10, -10),  # Where the light is pointed
        color='red',  # Color of the light
        intensity=1.0,  # Brightness of the light
        light_type='camera light'  # 'camera light' follows the camera
    )
    plotter.add_light(custom_light)

    # Add a coordinate axes indicator
    plotter.add_axes()

    # Enable shadow rendering (may impact performance)
    plotter.enable_shadows()

    # Start the interactive rendering window
    plotter.show()


def main():
    """
    Main entry point for loading a 3D scalar volume, extracting an isosurface,
    optionally smoothing it, and visualizing the result.
    """
    # 1. Load the scalar volume from the NIfTI file (e.g., 'brain-map.nii.gz')
    # Get relative pathS
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "Dataset", "brain-map.nii.gz")

    spacing = (1.0, 1.0, 1.0)  # Adjust if the dataset has different voxel sizes
    origin = (0.0, 0.0, 0.0)  # The (x, y, z) origin for the volume

    grid, data = load_scalar_volume(file_path, spacing, origin)

    # 2. Determine an iso-value for surface extraction
    #    Here, we use the 90th percentile of the volume's intensity distribution
    iso_value = np.percentile(data, 90)
    print(f"Using iso_value: {iso_value:.3f}")

    # 3. Extract the 3D isosurface using the chosen iso_value
    iso_surface = extract_iso_surface(grid, iso_value)

    # 3.1. Optionally smooth the extracted surface (300 iterations, relaxation factor = 1)
    smoothed_surface = iso_surface.smooth(n_iter=300, relaxation_factor=1)

    # 4. Visualize the (non-smoothed) or smoothed surface
    # Note: the line below uses the original iso_surface instead of the smoothed_surface.
    # If you want to see the smoothed version, replace iso_surface with smoothed_surface.
    visualize_3d_model(iso_surface)


if __name__ == "__main__":
    main()
