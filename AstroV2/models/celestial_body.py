import numpy as np
from constants.astrodata import get_ephermis_data, get_obj, tc2array


class CelestialBody:

    def __init__(self, name: str, cb: dict, times: int, frame: str='ECLIPJ2000', observer: str='SUN') -> None:
        """
        CelestialBody constructor.

        Parameters:
            name (str): The name of the celestial body.
            cb (dict): A dictionary containing the celestial body's properties.
            times (int): The number of times to calculate the ephemeris for.
            frame (str): The reference frame to use. Defaults to 'ECLIPJ2000'.
            observer (str): The observer to use. Defaults to 'SUN'.

        Returns:
            None
        """
        ids, names, tcs_sec, tcs_cal = get_obj()
        times = tc2array(tcs_sec[0], times)
        self.name = name
        name = name + ' BARYCENTER'
        self.cb = cb
        self.rs_list = get_ephermis_data(name, times, frame, observer)

    def ode_solver(self):
        return None, None, np.squeeze(self.rs_list)

    