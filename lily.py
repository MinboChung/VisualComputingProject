import vtk
import os

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

# 读取数据集
reader_fa = read_file(file_path_fa)
# reader_tensor = read_file(file_path_tensor)
# reader_vector = read_file(file_path_vector)

# 获取数据范围并设定合适的 iso_value
scalar_range_fa = reader_fa.GetOutput().GetScalarRange()
iso_value_fa = 0.5  # 适合 FA 数据范围
print(f"Scalar Range (fa): {scalar_range_fa}, Using iso_value: {iso_value_fa}")

# Marching Cubes 提取 FA 数据的等值面
marching_cubes_fa = vtk.vtkMarchingCubes()
marching_cubes_fa.SetInputConnection(reader_fa.GetOutputPort())
marching_cubes_fa.SetValue(0, iso_value_fa)
marching_cubes_fa.ComputeNormalsOn()
marching_cubes_fa.Update()

decimate = vtk.vtkDecimatePro()
decimate.SetInputConnection(marching_cubes_fa.GetOutputPort())
decimate.SetTargetReduction(0.1)  # 10% 的面减少，防止丢失主要结构
decimate.PreserveTopologyOn()

cleaner = vtk.vtkCleanPolyData()
cleaner.SetInputConnection(decimate.GetOutputPort())
cleaner.Update()

connectivity = vtk.vtkConnectivityFilter()
connectivity.SetInputConnection(cleaner.GetOutputPort())
connectivity.SetExtractionModeToLargestRegion()

smoother = vtk.vtkSmoothPolyDataFilter()
smoother.SetInputConnection(connectivity.GetOutputPort())
smoother.SetNumberOfIterations(30)  
smoother.SetRelaxationFactor(0.1)  
smoother.FeatureEdgeSmoothingOn()

# 设置 FA 数据的渲染
mapper_fa = vtk.vtkPolyDataMapper()
mapper_fa.SetInputConnection(marching_cubes_fa.GetOutputPort())
mapper_fa.ScalarVisibilityOff()

actor_fa = vtk.vtkActor()
actor_fa.SetMapper(mapper_fa)
actor_fa.GetProperty().SetColor(1.0, 0.8, 0.6)  # FA 数据颜色

# 设置 tensor 数据的渲染
mapper_tensor = vtk.vtkPolyDataMapper()
# 如果 tensor 数据需要特定的处理，可以在这里添加相应的处理代码
actor_tensor = vtk.vtkActor()
actor_tensor.SetMapper(mapper_tensor)
actor_tensor.GetProperty().SetColor(0.0, 1.0, 0.0)  # 设置颜色

# 设置 vector 数据的渲染
mapper_vector = vtk.vtkPolyDataMapper()
# 如果 vector 数据需要特定的处理，可以在这里添加相应的处理代码
actor_vector = vtk.vtkActor()
actor_vector.SetMapper(mapper_vector)
actor_vector.GetProperty().SetColor(0.0, 0.0, 1.0)  # 设置颜色

# 3D 渲染设置
renderer = vtk.vtkRenderer()
render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

# 视角调整
renderer.AddActor(actor_fa)
renderer.AddActor(actor_tensor)
renderer.AddActor(actor_vector)
renderer.SetBackground(0.1, 0.1, 0.1)
renderer.ResetCamera()
renderer.GetActiveCamera().Zoom(1.5)




# 渲染并交互
render_window.Render()
interactor.Start()
