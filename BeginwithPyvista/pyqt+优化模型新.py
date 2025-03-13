import os
import nibabel as nib
import numpy as np
import pyvista as pv
import scipy.ndimage
from joblib import Parallel, delayed
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QSlider, QLabel, QComboBox
from pyvistaqt import QtInteractor


class VolumeRenderingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Volume Rendering Interface")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        # Load Data
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, "Dataset", "brain-map.nii.gz")
        nii_map = nib.load(file_path)
        data_map = nii_map.get_fdata()

        scaled_data = scipy.ndimage.zoom(data_map, (2, 2, 2), order=1, mode='nearest')
        scaled_dims = scaled_data.shape

        self.resampled_grid = pv.ImageData(dimensions=scaled_dims, spacing=(0.5, 0.5, 0.5))
        self.resampled_grid.point_data["values"] = scaled_data.flatten(order="F")

        initial_contours = self.resampled_grid.contour([np.percentile(scaled_data, 90)])
        self.smoothed_model = Parallel(n_jobs=-1)(delayed(initial_contours.smooth)(n_iter=2000) for _ in range(1))[0]
        self.smoothed_model.compute_normals(cell_normals=True, point_normals=True, auto_orient_normals=True)

        self.current_height = scaled_dims[2] // 2
        self.clipped_model = self.smoothed_model.clip(normal=[0, 0, 1], origin=[0, 0, self.current_height])

        # PyVista Qt Widget
        self.plotter = QtInteractor(self)
        self.layout.addWidget(self.plotter.interactor)

        self.light = pv.Light(position=(1, 1, 1), intensity=0.8, light_type="headlight")
        self.plotter.add_light(self.light)
        self.model_actor = self.plotter.add_mesh(self.clipped_model, color='gray', smooth_shading=True, specular=0.2,
                                                 name="clipped_model")

        # Light Intensity Slider
        self.light_slider = QSlider()
        self.light_slider.setOrientation(1)
        self.light_slider.setMinimum(1)
        self.light_slider.setMaximum(20)
        self.light_slider.setValue(8)
        self.light_slider.valueChanged.connect(self.update_light_intensity)
        self.layout.addWidget(QLabel("Light Intensity"))
        self.layout.addWidget(self.light_slider)

        # Model Color Dropdown
        self.color_dropdown = QComboBox()
        self.color_dropdown.addItems(["gray", "blue", "red", "green", "white"])
        self.color_dropdown.currentIndexChanged.connect(self.update_model_color)
        self.layout.addWidget(QLabel("Model Color"))
        self.layout.addWidget(self.color_dropdown)

        # CT Mode Toggle Button
        self.ct_button = QPushButton("Toggle CT Mode")
        self.ct_button.clicked.connect(self.toggle_ct_mode)
        self.layout.addWidget(self.ct_button)

        # CT Slice Height Slider
        self.ct_slider = QSlider()
        self.ct_slider.setOrientation(1)
        self.ct_slider.setMinimum(0)
        self.ct_slider.setMaximum(scaled_dims[2])
        self.ct_slider.setValue(self.current_height)
        self.ct_slider.valueChanged.connect(self.update_clip)
        self.layout.addWidget(QLabel("CT Slice Height"))
        self.layout.addWidget(self.ct_slider)

        self.plotter.show()

    def update_light_intensity(self, value):
        self.light.intensity = value / 10.0
        self.plotter.render()

    def update_model_color(self, index):
        colors = ["gray", "blue", "red", "green", "white"]
        new_color = colors[index]
        self.plotter.remove_actor("clipped_model", reset_camera=False)
        self.model_actor = self.plotter.add_mesh(self.clipped_model, color=new_color, smooth_shading=True,
                                                 name="clipped_model")
        self.plotter.render()

    def toggle_ct_mode(self):
        pv.global_theme.allow_empty_mesh = True  # Allow empty meshes to avoid crashes
        self.plotter.remove_actor("ct_slice", reset_camera=False)
        ct_slice = self.resampled_grid.slice(normal=[0, 0, 1], origin=[0, 0, max(1, min(self.current_height,
                                                                                        self.resampled_grid.dimensions[
                                                                                            2] - 1))])
        if ct_slice.n_points == 0:
            print("Warning: CT slice is empty, skipping visualization.")
            return
        self.plotter.add_mesh(ct_slice, cmap="gray", opacity=1.0, name="ct_slice")
        self.plotter.render()

    def update_clip(self, value):
        self.current_height = value
        self.plotter.remove_actor("clipped_model", reset_camera=False)
        self.clipped_model = self.smoothed_model.clip(normal=[0, 0, 1], origin=[0, 0, self.current_height])
        self.model_actor = self.plotter.add_mesh(self.clipped_model, color='gray', smooth_shading=True,
                                                 name="clipped_model")
        self.plotter.render()


if __name__ == "__main__":
    app = QApplication([])
    window = VolumeRenderingApp()
    window.show()
    app.exec_()