import pyvista as pv
import numpy as np

if __name__=="__main__":
    # Define grid dimensions
    dims = (30, 30, 30)  # 30x30x30 grid
    # Create a uniform grid
    grid = pv.ImageData()
    grid.dimensions = dims
    grid.spacing = (1, 1, 1)  # Set voxel spacing
    
    # Extract points from the grid
    points = pv.PolyData(grid.points)  # Convert ImageData to PointSet
    
    # Assign a scalar field (distance from center as example)
    center = np.array([15, 15, 15])
    scalars = np.linalg.norm(points.points - center, axis=1) - 10  # Example scalar field
    points["values"] = scalars  # Assign to points
    
    # Convert to tetrahedral mesh using Delaunay 3D
    tetra_mesh = points.delaunay_3d()
    
    # Extract the isosurface
    isosurface = tetra_mesh.contour(isosurfaces=[0.0])
    
    # Visualize the result
    isosurface.plot()