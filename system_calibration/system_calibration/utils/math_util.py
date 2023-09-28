import numpy as np

"""
Wrap around radian angle between provided range, and
the wrap around includes range[0] and excludes range[1]
Params:
    rad, float, radian angle
    range, optional, the range to wrap around
Returns:
    Wrapped around angle in radian
"""
def wrap_around_rad(rad, range=(-np.pi, np.pi)):
    # safe to assume that the wrap around range should
    # be within 2 pi
    assert(range[1] - range[0] == 2 * np.pi)
    while rad >= range[1]:
        rad -= 2 * np.pi 
    while rad < range[0]:
        rad += 2 * np.pi 
    return rad

    
    