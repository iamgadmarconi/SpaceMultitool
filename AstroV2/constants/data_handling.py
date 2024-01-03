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

