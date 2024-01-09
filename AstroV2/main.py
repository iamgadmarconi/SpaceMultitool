import numpy as np
from physics.engine import OrbitPropagator
from viewer.viewer import Viewer
from constants.constants import Constants as C
from constants.astrodata import get_obj, tc2array, get_ephermis_data
from constants.math_utils import get_coes_from_tle
from models.celestial_body import CelestialBody


if __name__ == "__main__":
    # test()

    mercury = CelestialBody('MERCURY', C.body_data['Mercury'], 10000)
    venus = CelestialBody('VENUS', C.body_data['Venus'], 10000)
    earth = CelestialBody('EARTH', C.body_data['Earth'], 10000)
    mars = CelestialBody('MARS', C.body_data['Mars'], 10000)
    jupiter = CelestialBody('JUPITER', C.body_data['Jupiter'], 10000)
    # print(earth.rs_list)
    

    viewer = Viewer([mercury, venus, mars, jupiter], 'static')    

    # ids, names, tcs_sec, tcs_cal = get_obj()

    # names = [f for f in names if 'BARYCENTER' in f] 

    # print(tcs_sec[0])
    # times = tc2array(tcs_sec[0], 10000)

    # rs = []

    # for name in names: 
    #     print(name, times)
    #     rs.append(get_ephermis_data(name, times, 'ECLIPJ2000', 'SUN'))
    
    # viewer = Viewer()
    # viewer.plot_n_orbits(rs)
    perturbations = {'J2': True,
                    'J2,2': False, 
                    'drag': True, 
                    'lunar': False,
                    'srp': False,
                    'relativity': False}
     
    # stop_conditions = {'max_alt': 7000e3,
    #                     'min_alt': 600e3,
    #                     'deorbit': False}

    # r_mag0 = C.body_data['Earth']['radius'] + 1500000
    # v_mag0 = np.sqrt(C.body_data['Earth']['mu'] / r_mag0)

    # r0 =[r_mag0, 0, 0]
    # v0 = [0, v_mag0, 0]
    # state0 = r0 + v0
    # # state0_coes = get_coes_from_tle('CATNR', '25544', C.body_data['Earth']['mu'])
    state0_coes = [6800e3, 0, np.radians(45), np.radians(100), np.radians(250), np.radians(45)]

    # r_mag1 = C.body_data['Earth']['radius'] + 3000000
    # v_mag1 = np.sqrt(C.body_data['Earth']['mu'] / r_mag1) * 1.3

    # r1 = [r_mag1, 0, 0]
    # v1 = [0, v_mag1, 0.3]
    # state1 = r1 + v1
    # # state1_coes = get_coes_from_tle('CATNR', '27386', C.body_data['Earth']['mu'])
    # state1_coes = [42164e3, 0, np.radians(0), np.radians(45), np.radians(0), np.radians(0)]

    cb = C.body_data['Earth']

    tspan = 100000

    sat1 = OrbitPropagator(state0_coes, tspan, 100, cb, 'ISS', True, perturbations)
    # sat2 = OrbitPropagator(state1_coes, tspan, 100, cb, 'EnviSat', True)
    plot = Viewer([sat1], 'static')
