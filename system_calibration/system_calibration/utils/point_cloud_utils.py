import numpy as np
import pcl
from scipy.spatial import KDTree
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def numpy_to_pcd(np_array, filename):
    """
    Save a numpy (n,3) array to .pcd file

    :param np_array: (n,3) shape numpy array
    :param filename: string, name of the output .pcd file
    """

    # Convert the numpy array to PointCloud
    point_cloud = pcl.PointCloud(np_array.astype(np.float32))

    # Save to .pcd file
    pcl.save(point_cloud, filename)

def fit_plane(points):
    """
    Fit a plane to a set of 3D points using SVD.
    
    Parameters:
    - points: Nx3 numpy array of 3D points
    
    Returns:
    - normal: Normal vector of the plane
    - d: Distance of the plane from the origin
    """
    # Calculate centroid of the points
    centroid = points.mean(axis=0)
    
    # Use SVD to fit the plane
    _, _, vh = np.linalg.svd(points - centroid)
    normal = vh[2, :]
   
    return normal

def distance_to_plane(point, normal, d):
    """
    Calculate the distance from a point to a plane.
    
    Parameters:
    - point: 3D point
    - normal: Normal vector of the plane
    - d: Distance of the plane from the origin
    
    Returns:
    - distance: Signed distance to the plane
    """
    return abs(np.dot(point, normal) + d)

def plane_fitting(point, point_cloud, radius, kdtree=None, fit_std_thres=0.02):
    """
    Fit a plane to the points within a radius around a query point 
    and return the standard deviation of distances to the plane.
    
    Parameters:
    - point: 3D query point
    - kdtree: constructed kdtree 
    - radius: Search radius
    
    Returns:
    - std_dev: Standard deviation of distances to the fitted plane
    """

    def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0: 
           return v
        return v / norm

    if kdtree is None:
        kdtree = KDTree(point_cloud)
    indices = kdtree.query_ball_point(point, radius)
    
    # Extract neighbors from the point cloud
    neighbors = point_cloud[indices]
    if len(neighbors) < 5:
        return None
    
    # Fit a plane to the neighbors
    normal = fit_plane(neighbors)
    if normal.dot(point) < 0:
        normal = -normal
    # Calculate distance of the plane from the origin
    d = -neighbors.mean(axis=0).dot(normal)
    
    # Calculate distances of neighbors to the plane
    distances = [distance_to_plane(p, normal, d) for p in neighbors]
    
    if np.std(distances) > fit_std_thres:
        return None
    
    return normal 

def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def vis_points_with_normal(points, normals, show_origin=False, show_point_direction=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if show_origin:
        ax.quiver(0, 0, 0, 1, 0, 0, color='r', length=0.2)
        ax.quiver(0, 0, 0, 0, 1, 0, color='g', length=0.2)
        ax.quiver(0, 0, 0, 0, 0, 1, color='b', length=0.2)

    for i, (point, normal) in enumerate(zip(points, normals)):
        # Define the axis lengths
        axis_length = 0.1
        # Plot x, y, z-axes
        ax.quiver(*point, *normal, color='r', length=axis_length)
        if show_point_direction:
            ax.quiver(*point, *point, color='b', length=axis_length)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    set_axes_equal(ax)
    plt.show()

