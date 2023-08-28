import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

TARGET_CENTER = [3, 0, 0.8]

def quat_to_rot(qx, qy, qz, qw):
    # Convert quaternion to rotation matrix
    return np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])

def lerp(start, end, alpha):
    return (1 - alpha) * np.array(start) + alpha * np.array(end)

def point_to_axis_quaternion(position, target_point, axis_to_align):
    direction = np.array(target_point) - np.array(position[0:3])
    direction = direction / np.linalg.norm(direction)
    axis = np.cross(axis_to_align, direction)
    axis_norm = np.linalg.norm(axis)
    
    if axis_norm < 1e-6:
        return [0., 0., 0., 1.]  # already aligned
    
    axis = axis / axis_norm
    angle = np.arccos(np.dot(axis_to_align, direction))
    quaternion = R.from_rotvec(axis * angle).as_quat()
    return quaternion.tolist()

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

def vis_traj(poses):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.quiver(0, 0, 0, 1, 0, 0, color='r', length=0.2)
    ax.quiver(0, 0, 0, 0, 1, 0, color='g', length=0.2)
    ax.quiver(0, 0, 0, 0, 0, 1, color='b', length=0.2)

    for i, pose in enumerate(poses):
        x, y, z, qx, qy, qz, qw = pose
        R = quat_to_rot(qx, qy, qz, qw)

        origin = np.array([x, y, z])

        # Define the axis lengths
        axis_length = 0.1

        # Plot x, y, z-axes
        ax.quiver(origin[0], origin[1], origin[2], R[0, 0], R[1, 0], R[2, 0], color='r', length=axis_length, label="x-axis" if i == "0" else "")
        ax.quiver(origin[0], origin[1], origin[2], R[0, 1], R[1, 1], R[2, 1], color='g', length=axis_length, label="y-axis" if i == "0" else "")
        ax.quiver(origin[0], origin[1], origin[2], R[0, 2], R[1, 2], R[2, 2], color='b', length=axis_length, label="z-axis" if i == "0" else "")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    set_axes_equal(ax)
    plt.show()


def generate_traj():
    target_point = TARGET_CENTER 
    poses_num = 100
    key_points = {
        0: [0.7, -1, 1., 0., 0., 0., 1.],
        25: [0.7, 1., 1, 0., 0., 0., 1.],
        50: [0.7, 1., 0.4, 0., 0., 0., 1.],
        75: [0.7, -1., 0.4, 0., 0., 0., 1.],
        99: [0.7, -1., 1, 0., 0., 0., 1.]
    }
    
    poses = [] 
    indices = list(key_points.keys())
    for i in range(poses_num):
        idx1 = max([x for x in indices if x <= i])
        idx2 = min([x for x in indices if x >= i])
        
        if idx1 == idx2:
            pose = key_points[idx1]
        else:
            alpha = (i - idx1) / (idx2 - idx1)
            pose = lerp(key_points[idx1], key_points[idx2], alpha).tolist()
        
        quaternion = point_to_axis_quaternion(pose, target_point, [1, 0, 0])
        pose[3:7] = quaternion
        poses.append(pose)

    return poses 
 
def main():
    # generate trajectories
    poses = generate_traj()       

    # visualize the traj
    vis_traj(poses)

    # dump that a robot plan file
    poses_fmt = dict() 
    for i, pose in enumerate(poses):
        poses_fmt[i] = {
            'tt_angle': 0.0,
            'track_position': 0.0,
            'robot_planning_pose': pose + ["base_link", "link_6", "autel_small_target"]
        }
    with open('.poses.yaml', 'w') as outfile:
        yaml.dump(poses_fmt, outfile)


if __name__ == '__main__':
    main()
