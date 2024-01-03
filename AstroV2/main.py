import numpy as np
from physics.engine import OrbitPropagator
from viewer.viewer import Viewer
from constants.constants import Constants as C

if __name__ == "__main__":

    r_mag0 = C.body_data['Earth']['radius'] + 1500000
    v_mag0 = np.sqrt(C.body_data['Earth']['mu'] / r_mag0)

    r0 =[r_mag0, 0, 0]
    v0 = [0, v_mag0, 0]

    r_mag1 = C.body_data['Earth']['radius'] + 3000000
    v_mag1 = np.sqrt(C.body_data['Earth']['mu'] / r_mag1) * 1.3

    r1 = [r_mag1, 0, 0]
    v1 = [0, v_mag1, 0.3]

    cb = C.body_data['Earth']

    sat1 = OrbitPropagator(r0, v0, 100000, 10, cb, 'ISS')
    sat2 = OrbitPropagator(r1, v1, 100000, 10, cb, 'Apollo 11')
    plot = Viewer([sat1, sat2])

    plot.plot_animated(5)
