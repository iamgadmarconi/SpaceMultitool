import requests

def get_tle(query: str, value: str, format='tle') -> list:
    """
    Fetch TLE data from Celestrak.

    Parameters:
        query (str): The type of query, e.g., 'CATNR', 'NAME', 'GROUP'.
        value (str): The value for the query, e.g., satellite catalog number, satellite name, group name.
        format (str): Format of the output, default is 'tle'.

    Returns:
        str: The TLE data as a string, or an error message if unsuccessful.
    """
    base_url = "https://celestrak.org/NORAD/elements/gp.php"
    params = {
        query: value,
        'FORMAT': format
    }
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.text.strip().split('\n')[-2:]
    
    except requests.exceptions.HTTPError as err:
        return f"HTTP error occurred: {err}"
    
    except Exception as err:
        return f"Other error occurred: {err}"

def perturbations():
    """
    Perturbations to use in the orbit propagator.

    Returns:
        dict: A dictionary containing the perturbations to use.
            J2 (bool): Perturbations due to the J2 term.
            J2,2 (bool): Perturbations due to the J2,2 term.
            drag (bool): Perturbations due to atmospheric drag.
            lunar (bool): Perturbations due to the moon.
            srp (bool): Perturbations due to solar radiation pressure.
            relativity (bool): Perturbations due to relativity.
    """
    return {'J2': False,
            'J2,2': False, 
            'drag': False, 
            'lunar': False,
            'srp': False,
            'relativity': False}

def stop_conditions():
    """
    Stop conditions to use in the orbit propagator.

    Returns:
        dict: A dictionary containing the stop conditions to use.
            max_alt (float): The maximum altitude.
            min_alt (float): The minimum altitude.
            deorbit (bool): Whether to stop when the satellite deorbits.
    """
    return {'max_alt': None,
            'min_alt': None,
            'deorbit': False}
    


