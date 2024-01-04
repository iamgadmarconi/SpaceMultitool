import numpy as np
from physics.engine import OrbitPropagator
from viewer.viewer import Viewer
from constants.constants import Constants as C
from constants.math_utils import get_coes_from_tle


if __name__ == "__main__":

    perturbations = {'J2': True,
                    'J2,2': False, 
                    'drag': False, 
                    'lunar': False,
                    'srp': False,
                    'relativity': False}

    r_mag0 = C.body_data['Earth']['radius'] + 1500000
    v_mag0 = np.sqrt(C.body_data['Earth']['mu'] / r_mag0)

    r0 =[r_mag0, 0, 0]
    v0 = [0, v_mag0, 0]
    state0 = r0 + v0
    # state0_coes = get_coes_from_tle('CATNR', '25544', C.body_data['Earth']['mu'])
    state0_coes = [6800e3, 0, np.radians(45), np.radians(100), np.radians(250), np.radians(45)]

    r_mag1 = C.body_data['Earth']['radius'] + 3000000
    v_mag1 = np.sqrt(C.body_data['Earth']['mu'] / r_mag1) * 1.3

    r1 = [r_mag1, 0, 0]
    v1 = [0, v_mag1, 0.3]
    state1 = r1 + v1
    # state1_coes = get_coes_from_tle('CATNR', '27386', C.body_data['Earth']['mu'])
    state1_coes = [42164e3, 0, np.radians(0), np.radians(45), np.radians(0), np.radians(0)]

    cb = C.body_data['Earth']

    sat1 = OrbitPropagator(state0_coes, 100000, 100, cb, 'ISS', True, perturbations)
    sat2 = OrbitPropagator(state1_coes, 100000, 100, cb, 'EnviSat', True, perturbations)
    plot = Viewer([sat1, sat2], 'animated')
