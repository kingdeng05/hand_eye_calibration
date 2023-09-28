import numpy as np
from collections import defaultdict

class SystemSimulator(object):
    def __init__(self):
        self._components = dict() 
        self._calibration = defaultdict(dict) 

    @property
    def components(self):
        return self._components

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
    def add_component(self, name, component, name_base, base_tf_comp):
        assert(name not in self._components)
        self.components[name] = component 
        # add calibration
        self._calibration[name][name_base] = base_tf_comp
        self._calibration[name_base][name] = np.linalg.inv(base_tf_comp)

    """
    Get transform from source to target by search in the graph
    Params:
        name_source, str
        name_target, str
    Returns:
        4x4 transformation matrix 
    """
    def get_transform(self, name_source, name_target):
        def search_dfs(calib_graph, name_source, name_target, visited):
            if name_source == name_target:
                return self._components[name_target].frame 
            visited.add(name_source)
            for n_t, tf in calib_graph.get(name_source, {}).items():
                if n_t not in visited:
                    tf_ret = search_dfs(calib_graph, n_t, name_target, visited)
                    if tf_ret is not None:
                        return tf_ret @ tf @ self._components[name_source].frame
            return None
        result = search_dfs(self._calibration, name_source, name_target, set())
        if result is None:
            raise RuntimeError(f"Calibration from {name_source} to {name_target} not found")
        return result
