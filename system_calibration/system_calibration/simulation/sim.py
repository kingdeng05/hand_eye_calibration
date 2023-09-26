import numpy as np

from py_kinetic_backend import Pose3, Rot3, PinholeCameraCal3DS2, Cal3DS2

IMG_WIDTH = 2048 
IMG_HEIGHT = 1536 
FOCAL_LENGTH = {
    "6mm": 1739,
    "4mm": 1159
}

def create_calib_gt():
    # rzryrx first rotate around z axis, then y axis, lastly x axis.
    calib_rot_gt = Rot3.rzryrx([-np.pi/2, 0, -np.pi/2])
    calib_t_gt = np.array([0.13, 0, 0])
    calib_gt = Pose3(calib_rot_gt, calib_t_gt).matrix()
    return calib_gt

def create_t2w_gt():
    calib_rot_gt = Rot3.rzryrx([-np.pi/2, 0, -np.pi/2])
    calib_t_gt = np.array([3, 0, 0.8])
    t2w = Pose3(calib_rot_gt, calib_t_gt).matrix()
    return t2w 

def create_cube_t2w_gt():
    calib_rot_gt = Rot3.rzryrx([-np.pi/2, 0, -np.pi/2])
    calib_t_gt = np.array([3.8, 0, 0.143])
    t2w = Pose3(calib_rot_gt, calib_t_gt).matrix()
    return t2w 

def create_intrinsic_distortion(focal_length="6mm", distortion_model="Cal3DS2"):
    focal_length_xy = FOCAL_LENGTH[focal_length]
    if distortion_model == "Cal3DS2":
        return np.array([focal_length_xy, focal_length_xy, 0, IMG_WIDTH/2, IMG_HEIGHT/2, 0, 0, 0, 0])
    elif distortion_model == "Cal3Rational":
        return np.array([focal_length_xy, focal_length_xy, IMG_WIDTH/2, IMG_HEIGHT/2, 0, 0, 0, 0, 0, 0, 0, 0])
    elif distortion_model == "Cal3_S2":
        return np.array([focal_length_xy, focal_length_xy, IMG_WIDTH/2, IMG_HEIGHT/2])
    else:
        raise NotImplementedError(f"Model {distortion_model} isn't supported!")

class VisibleCamera(object):
    def __init__(self, intrinsic):
        self.intrinsic = intrinsic
         
    def project(self, cam_pose, target, img_size):
        camera = PinholeCameraCal3DS2(Pose3(cam_pose), Cal3DS2(self.intrinsic)) 
        pts_2d = []
        pts_3d = []
        for id in target.use_ids:
            pts_target = target.get_pts_3d_by_id(id)
            view_dir = Pose3(cam_pose).transform_to(pts_target.mean(axis=0))
            # calculate the z axis of the face
            pose_vec = target.frames[id] # remember this is from target base to target face
            tf_base2face = Pose3(Rot3.rzryrx(pose_vec[:3]), pose_vec[3:]).matrix()
            face_vec_cam = (np.linalg.inv(tf_base2face @ cam_pose)[:3, :3].dot(np.array([0, 0, 1]).reshape(-1, 1))).flatten()
            if self.is_face_visible(face_vec_cam, view_dir):
                for pt_target in pts_target:
                    pt_proj = camera.project(pt_target)
                    if pt_proj[0] >= 0 and pt_proj[0] < img_size[0] and \
                       pt_proj[1] >= 0 and pt_proj[1] < img_size[1]:
                        pts_2d.append(pt_proj) 
                        pts_3d.append(pt_target)
        return np.array(pts_2d).reshape(-1, 2), np.array(pts_3d).reshape(-1, 3)
    
    @staticmethod
    def transform_pts(tf, pts):
        pts_tf = []
        for pt in pts:
            pts_tf.append(Pose3(tf).transform_from(pt))
        return np.array(pts_tf)
            
    @staticmethod
    def is_face_visible(normal, view_direction):
        # Normalize the normal and the view direction
        normal = normal / np.linalg.norm(normal)
        view_direction = view_direction / np.linalg.norm(view_direction)
        # Note: normal is actually pointing inward 
        # If the dot product is positive, the face is facing towards from the camera
        return np.dot(normal, view_direction) > 0


