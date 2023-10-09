import numpy as np

def is_plane_visible(normal, view_direction):
    # Normalize the normal and the view direction
    normal = normal / np.linalg.norm(normal)
    view_direction = view_direction / np.linalg.norm(view_direction)
    # Note: normal is actually pointing inward 
    # If the dot product is positive, the face is facing towards from the camera
    return np.dot(normal, view_direction) > 0