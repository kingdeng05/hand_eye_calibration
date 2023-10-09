import numpy as np

from .perception_component import PerceptionComponent
from ..target_components import ArucoCubeTarget, ArucoBoardTarget, CheckerboardTarget 
from .utils import is_plane_visible 
from ....utils import euler_vec_to_mat, transform_3d_pts


class Camera(PerceptionComponent):
    def __init__(self, camera, img_size):
        self._camera = camera
        self._img_size = img_size
        super().__init__()

    """
    Capture the image of target
    Params:
        cam_pose, 4x4 matrix, from camera to target
        target, targets or just numpy array 
    Returns:
        the 2d pts and 3d pts that are visible
    """
    def capture(self, cam_pose, target):
        pts_2d = []
        pts_3d = []
        if isinstance(target, ArucoCubeTarget): 
            pts_2d_all, pts_3d_all = [], []
            for id in target.use_ids:
                pts_target = target.get_pts_3d_by_id(id)
                tf_base2face = euler_vec_to_mat(target.frames[id], use_deg=False) # remember this is from target base to target face
                pts_target = transform_3d_pts(pts_target, tf_base2face)
                pts_2d, pts_3d = self._get_visible_pts(pts_target, tf_base2face @ cam_pose)
                pts_3d = transform_3d_pts(pts_3d, np.linalg.inv(tf_base2face))
                pts_2d_all.append(pts_2d)
                pts_3d_all.append(pts_3d)
            pts_2d_all = np.vstack(pts_2d_all)
            pts_3d_all = np.vstack(pts_3d_all)
            return pts_2d_all, pts_3d_all 
        elif isinstance(target, ArucoBoardTarget) or isinstance(target, CheckerboardTarget):
            pts_target = target.get_pts()
            return self._get_visible_pts(pts_target, cam_pose)
        elif isinstance(target, np.ndarray):
            # dummy projection 
            pts_cam = transform_3d_pts(target, np.linalg.inv(cam_pose))
            return self.pts_in_image(pts_cam) 
        else:
            raise NotImplementedError(f"TargetType {type(target)} is not supported yet!")

    def _get_visible_pts(self, pts_target, cam_pose):
        pts_cam = transform_3d_pts(pts_target, np.linalg.inv(cam_pose))
        view_dir = pts_cam.mean(axis=0) 
        # calculate the z axis of the face
        face_vec_cam = (np.linalg.inv(cam_pose)[:3, :3].dot(np.array([0, 0, 1]).reshape(-1, 1))).flatten()
        if not is_plane_visible(face_vec_cam, view_dir):
            pts_cam = [] # clear the points
        pts_img_valid, pts_cam_valid = self._pts_in_image(pts_cam) 
        pts_target_valid = transform_3d_pts(pts_cam_valid, cam_pose)
        return pts_img_valid, pts_target_valid

    def _pts_in_image(self, pts_cam):
        pts_2d = []
        pts_3d = []
        for pt_target in pts_cam:
            pt_proj = self._camera.project(pt_target)
            if pt_proj[0] >= 0 and pt_proj[0] < self._img_size[0] and \
               pt_proj[1] >= 0 and pt_proj[1] < self._img_size[1]:
                pts_2d.append(pt_proj) 
                pts_3d.append(pt_target)
        return np.array(pts_2d).reshape(-1, 2), np.array(pts_3d).reshape(-1, 3)
         
