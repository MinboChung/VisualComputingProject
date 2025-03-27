import vtk
import os

# 设置 NIfTI 文件的路径

file_path = os.path.abspath(os.path.join(".\CT_Philips", "CT_Philips.nii") )  # 确保路径正确
print("Full Path:", file_path)

# 创建 NIfTI 读取器
print(file_path)

reader = vtk.vtkNIFTIImageReader()
reader.SetFileName(file_path)
reader.Update()

# 创建一个简单的 2D 影像查看器
imageViewer = vtk.vtkImageViewer2()
imageViewer.SetInputConnection(reader.GetOutputPort())

# 创建窗口交互
interactor = vtk.vtkRenderWindowInteractor()
imageViewer.SetupInteractor(interactor)

# 渲染并启动交互
imageViewer.Render()
imageViewer.GetRenderer().ResetCamera()
interactor.Start()
