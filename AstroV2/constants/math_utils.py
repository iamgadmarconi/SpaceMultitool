import numpy as np
from datetime import datetime, timedelta
from constants.data_handling import get_tle


def coes2rv(coes: list, mu: float, deg=False) -> tuple:
    """
    Converts COES to RV.

    Parameters:
        coes (list): The COES to convert.
        mu (float): The gravitational parameter of the body.
        deg (bool): Whether the COES are in degrees or not. Defaults to False.
    Returns:
        tuple: The RV.
    """
    a, e, i, raan, argp, nu = coes

    if deg:
        i = np.deg2rad(i)
        raan = np.deg2rad(raan)
        argp = np.deg2rad(argp)
        nu = np.deg2rad(nu)

    tau = period(a, mu)
    T = tau / (2 * np.pi) * (nu - e * np.sin(nu)) # Time since perigee passage
    M = mean_anomaly_from_time(T, tau)
    E = eccentric_anomaly(M, e)

    # Calculate r_norm
    r_norm = a * (1 - e**2) / (1 + e * np.cos(nu))

    r_perif = np.array([r_norm * np.cos(nu), r_norm * np.sin(nu), 0]) # Perifocal frame
    v_perif = np.array([-np.sin(nu), e + np.cos(nu), 0]) * np.sqrt(mu / a / (1 - e**2)) # Perifocal frame

    perif2eci = np.linalg.inv(eci2perif(raan, argp, i)) # Transformation matrix from perifocal to ECI

    r = np.dot(perif2eci, r_perif)
    v = np.dot(perif2eci, v_perif)

    return np.array(r), np.array(v)

def rv2coes(r: list, v: list, mu: float, deg=False) -> tuple:
    """
    Converts RV to COES.

    Parameters:
        r (list): The position vector.
        v (list): The velocity vector.
        mu (float): The gravitational parameter of the body.
    Returns:
        tuple: The COES.
    """
    h = np.cross(r, v)
    n = np.cross([0, 0, 1], h)
    e = np.cross(v, h) / mu - r / np.linalg.norm(r)
    a = 1 / (2 / np.linalg.norm(r) - np.linalg.norm(v) ** 2 / mu)
    i = np.arccos(h[2] / np.linalg.norm(h))
    raan = np.arccos(n[0] / np.linalg.norm(n))
    argp = np.arccos(np.dot(n, e) / (np.linalg.norm(n) * np.linalg.norm(e)))
    nu = np.arccos(np.dot(e, r) / (np.linalg.norm(e) * np.linalg.norm(r)))

    if deg:
        i = np.rad2deg(i)
        raan = np.rad2deg(raan)
        argp = np.rad2deg(argp)
        nu = np.rad2deg(nu)

    return np.array([a, np.linalg.norm(e), i, raan, argp, nu])

def eci2perif(raan: float, aop: float, i: float) -> np.array:
    """
    Calculates the transformation matrix from ECI to perifocal frame.

    Parameters:
        raan (float): The right ascension of the ascending node.
        aop (float): The argument of perigee.
        i (float): The inclination.
    Returns:
        np.array: The transformation matrix.
    """
    return np.array([[-np.sin(raan) * np.cos(i) * np.sin(aop) + np.cos(raan) * np.cos(aop), -np.sin(raan) * np.cos(i) * np.cos(aop) - np.cos(raan) * np.sin(aop), np.sin(raan) * np.sin(i)],
                     [np.cos(raan) * np.cos(i) * np.sin(aop) + np.sin(raan) * np.cos(aop), np.cos(raan) * np.cos(i) * np.cos(aop) - np.sin(raan) * np.sin(aop), -np.cos(raan) * np.sin(i)],
                     [np.sin(i) * np.sin(aop), np.sin(i) * np.cos(aop), np.cos(i)]])

def period(a: float, mu: float) -> float:
    """
    Calculates the period of an orbit.

    Parameters:
        a (float): The semi-major axis of the orbit.
        mu (float): The gravitational parameter of the body.
    Returns:
        float: The period of the orbit.
    """
    return 2 * np.pi * np.sqrt(a ** 3 / mu)

def mean_anomaly_from_time(T: float, P: float) -> float:
    """
    Calculate the Mean Anomaly given time since perigee passage and orbital period.

    Parameters:
        T (float): Time since perigee passage in seconds.
        P (float): Orbital period in seconds.

    Returns:
    float: Mean Anomaly in radians.
    """
    # Mean motion (rad/s)
    n = 2 * np.pi / P

    # Calculate Mean Anomaly
    M = n * T

    # Normalize M to be within the range of 0 to 2*pi
    M = np.mod(M, 2 * np.pi)

    return M

def eccentric_anomaly(M: float, e: float, tolerance=1e-6) -> float:
    """
    Calculate the Eccentric Anomaly given Mean Anomaly (M) and Eccentricity (e).

    Parameters:
        M (float): Mean Anomaly in radians.
        e (float): Eccentricity of the orbit.
        tolerance (float): Tolerance for the Newton-Raphson iterative method.

    Returns:
        float: Eccentric Anomaly in radians.
    """
    # Initial guess for E
    E = M if e < 0.8 else np.pi

    # Newton-Raphson iterative method
    while True:
        delta = E - e * np.sin(E) - M
        if abs(delta) < tolerance:
            break
        E -= delta / (1 - e * np.cos(E))

    return E

def E2NU(E: float, e: float) -> float:
    """
    Calculate the True Anomaly given Eccentric Anomaly (E) and Eccentricity (e).

    Parameters:
        E (float): Eccentric Anomaly in radians.
        e (float): Eccentricity of the orbit.

    Returns:
        float: True Anomaly in radians.
    """
    return 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))

def tle2coes(tle: str, mu: float) -> tuple:
    """
    Converts TLE to COES.

    Parameters:
        tle (str): The TLE to convert.
    """
    line1, line2 = tle
    inclination = float(line2[8:16])  # Degrees
    raan = float(line2[17:25])        # Degrees
    eccentricity = float('0.' + line2[26:33])
    arg_perigee = float(line2[34:42]) # Degrees
    mean_anomaly = float(line2[43:51])# Degrees
    mean_motion = float(line2[52:63]) # Revolutions per day

    mean_motion_rad = mean_motion * 2 * np.pi / (24 * 60 * 60)
    mean_anomaly_rad = np.radians(mean_anomaly)
    raan_rad = np.radians(raan)
    arg_perigee_rad = np.radians(arg_perigee)
    inclination_rad = np.radians(inclination)
    a = (mu / (mean_motion_rad**2)) ** (1./3.)

    E = eccentric_anomaly(mean_anomaly_rad, eccentricity)

    true_anomaly_rad = E2NU(E, eccentricity)

    # COEs: [semi-major axis, eccentricity, inclination, raan, argument of perigee, mean anomaly]
    return [a, eccentricity, inclination_rad, raan_rad, arg_perigee_rad, true_anomaly_rad]

def epoch(tle):
    """
    Returns the epoch of the TLE.

    Parameters:
        tle (str): The TLE to convert.
    """
    line1, line2 = tle
    epoch_str = line1[18:32]

    # Parse the year and day of year
    year = int("20" + epoch_str[:2])  # TLE uses two-digit year
    day_of_year = float(epoch_str[2:])

    # Convert to datetime
    start_of_year = datetime(year, 1, 1)
    epoch = start_of_year + timedelta(days=day_of_year - 1)

    return epoch

def get_coes_from_tle(query: str, value: str, mu: float, format='tle') -> tuple:
    """
    Fetch TLE data from Celestrak and convert to COES.

    Parameters:
        query (str): The type of query, e.g., 'CATNR', 'NAME', 'GROUP'.
        value (str): The value for the query, e.g., satellite catalog number, satellite name, group name.
        mu (float): The gravitational parameter of the body.
        format (str): Format of the output, default is 'tle'.

    Returns:
        tuple: The COES, or an error message if unsuccessful.
    """
    tle = get_tle(query, value, format)
    if len(tle) == 2:
        return tle2coes(tle, mu)
    else:
        return tle