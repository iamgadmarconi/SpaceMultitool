import ephem
import numpy as np


def get_heliocentric_coordinates(hlon_rad: float, hlat_rad: float, sun_distance: float) -> tuple:
    """
    Gets the heliocentric cartesian coordinates of a body relative to the Sun.

    Parameters:
        hlon_rad (float): The heliocentric longitude of the body in radians.
        hlat_rad (float): The heliocentric latitude of the body in radians.
        sun_distance (float): The distance from the Sun to the body in AU.

    Returns:
        tuple: The heliocentric cartesian coordinates of the body relative to the Sun.
    """
    x = sun_distance * np.cos(hlat_rad) * np.cos(hlon_rad)
    y = sun_distance * np.cos(hlat_rad) * np.sin(hlon_rad)
    z = sun_distance * np.sin(hlat_rad)
    
    return x, y, z

def compute_initial_position(body_name: str) -> np.array:
    """
    Computes the initial position of a body.

    Parameters:
        body_name (str): The name of the body.

    Returns:
        np.array: The initial position of the body.
    """
    # Create an observer object for the body
    if body_name.lower() == 'sun':
        return np.array([0, 0, 0])

    observer = ephem.Observer()
    observer.date = ephem.now()

    # For Earth, compute the Sun's position and invert it
    if body_name.lower() == 'earth':
        sun = ephem.Sun(observer)
        sun.compute(observer)
        earth_hlon = ephem.degrees(sun.hlon - ephem.pi)  # 180 degrees opposite
        earth_hlat = -sun.hlat                          # Inverse latitude
        earth_distance = sun.earth_distance             # Same distance
    else:
        # For other bodies
        body = getattr(ephem, body_name.capitalize())(observer)
        body.compute(observer)
        earth_hlon = body.hlon
        earth_hlat = body.hlat
        earth_distance = body.sun_distance

    pos = np.array(get_heliocentric_coordinates(earth_hlon, earth_hlat, earth_distance))
    return pos

def compute_initial_velocity(body_name: str, time_step=1000) -> np.array:
    """
    Computes the initial velocity of a body.

    Parameters:
        body_name (str): The name of the body.
        time_step (int): The time step to use for the calculation.

    Returns:
        np.array: The initial velocity of the body.
    """
    # Create an observer object for the body
    if body_name.lower() == 'sun':
        return np.array([0, 0, 0])

    observer1 = ephem.Observer()
    observer1.date = ephem.now()

    observer2 = ephem.Observer()
    observer2.date = observer1.date - ephem.second * time_step

    if body_name.lower() == 'earth':
        # For Earth, compute the Sun's position at two different times
        sun1 = ephem.Sun(observer1)
        sun1.compute(observer1)

        sun2 = ephem.Sun(observer2)
        sun2.compute(observer2)

        pos1 = np.array(get_heliocentric_coordinates(sun1.hlon, sun1.hlat, sun1.earth_distance))
        pos2 = np.array(get_heliocentric_coordinates(sun2.hlon, sun2.hlat, sun2.earth_distance))

    else:
        # For other bodies
        body = getattr(ephem, body_name.capitalize())()

        body.compute(observer1)
        pos1 = np.array(get_heliocentric_coordinates(body.hlon, body.hlat, body.sun_distance))

        body.compute(observer2)
        pos2 = np.array(get_heliocentric_coordinates(body.hlon, body.hlat, body.sun_distance))

    # Calculate the displacement vector
    displacement = pos2 - pos1

    # Calculate the initial velocity
    velocity = displacement / time_step

    return velocity
