import numpy as np
from collections import defaultdict

from .components import MovableComponent, PerceptionComponent, Camera, Target

class SystemSimulator(object):
    def __init__(self):
        self._components = dict() 
        self._calibration = defaultdict(dict) 
        self._direct_transform = defaultdict(list)
        self._moving = defaultdict(dict) 

    @property
    def components(self):
        return self._components

    @property
    def calibration(self):
        return self._calibration

    """
    Call the object with component name
    Params:
        name, str
    Returns:
        object
    """
    def __call__(self, name):
        return self._components[name] 

    """
    Get component by name
    Params:
        name, str, name of the component
    Returns:
        component, Component
    """
    def get_component(self, name):
        return self._components.get(name, None) 

    """
    Add component into the system simulator
    Params:
        name, str, name of the component
        component, Component
        name_base, str, name of the base frame that the component is rigidly attached to
        base_tf_comp, transformation from comp to base
    Returns
        None
    """
    def add_component(self, name, component, name_base, base_tf_comp, source_moving, base_moving):
        assert(name not in self._components)
        self.components[name] = component 
        # add calibration
        self._calibration[name][name_base] = base_tf_comp
        self._calibration[name_base][name] = np.linalg.inv(base_tf_comp)
        self._moving[name][name_base] = source_moving 
        self._moving[name_base][name] = base_moving 

    """
    Get transform from source to target by search in the graph
    Params:
        name_source, str
        name_target, str
    Returns:
        4x4 transformation matrix 
    """
    def get_transform(self, name_source, name_target, to_base=True):
        def search_dfs(calib_graph, name_source, name_target, visited):
            if name_source == name_target:
                return self._components[name_target].frame 
            visited.add(name_source)
            for n_t, tf in calib_graph.get(name_source, {}).items():
                if n_t not in visited:
                    tf_ret = search_dfs(calib_graph, n_t, name_target, visited)
                    if tf_ret is not None:
                        frame = self._components[name_source].frame
                        if self._moving[name_source][n_t]:
                            frame = np.linalg.inv(frame)
                            if n_t == name_target:
                                if (to_base and not self._moving[n_t][name_source]) or (not to_base and self._moving[n_t][name_source]):
                                    tf_ret = np.eye(4)
                                if (not to_base and not self._moving[n_t][name_source]):
                                    tf_ret = np.linalg.inv(tf_ret)
                        return tf_ret @ tf @ frame 
            return None
        result = search_dfs(self._calibration, name_source, name_target, set())
        if result is None:
            raise RuntimeError(f"Calibration from {name_source} to {name_target} not found")
        return result

    """
    Move the provided component
    Params:
        name, str, the component name
        val, the value to move
    Returns:
        None
    """
    def move(self, name: str, val):
        component = self.get_component(name)
        assert(isinstance(component, MovableComponent))
        component.move(val) 

    """
    Capture targets by perception components 
    Params:
        name, str, the component name
        target, optional, if empty(default) will capture all the targets in the system,
                otherwise specify any target name specifically
    Returns:
        Perception result 
    """
    def capture(self, name: str, targets=[]):
        component = self._components[name]
        assert(isinstance(component, PerceptionComponent))
        # get all target components
        if isinstance(component, Camera):
            pts_2d, pts_3d = [], []
            if len(targets) == 0:
                targets = dict() 
                for n, comp in self._components.items():
                    if isinstance(comp, Target):
                        targets[n] = comp
                for n, target in targets.items():
                    tf_cam2target = self.get_transform(name, n)
                    ret = component.capture(tf_cam2target, target) 
                    pts_2d.append(ret[0])
                    pts_3d.append(ret[1])
                    return np.vstack(pts_2d), np.vstack(pts_3d) 
            else:
                for t_n in targets:
                    assert(t_n in self._components)
                    tf_cam2target = self.get_transform(name, t_n)
                    return component.capture(tf_cam2target, self._components[t_n])
        else:
            raise NotImplementedError()

