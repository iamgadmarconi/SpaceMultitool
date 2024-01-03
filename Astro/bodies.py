import numpy as np
from astromodel_v1 import Body
from astrodata import compute_initial_position, compute_initial_velocity
from constants import Constants as C


body_data = [
            {
                'key': 0,
                'name': 'Sun',
                'mass': 1.989e30,
                'radius': 6.957e8,
                'position': [0, 0, 0],
                'velocity': [0, 0, 0],
                'acceleration': [0, 0, 0],
                'force': [0, 0, 0]
            },
            
            {
                'key': 1,
                'name': 'Earth',
                'mass': 5.972e24,
                'radius': 6.371e6,
                'position': [0, 1.49e11, 0],
                'velocity': [0, 0, 0],
                'acceleration': [0, 0, 0],
                'force': [0, 0, 0]
            },
            
            {
                'key': 2,
                'name': 'Moon',
                'mass': 7.34767309e22,
                'radius': 1.737e6,
                'position': [3.844e8, 0, 0],
                'velocity': [0, 1022, 0],
                'acceleration': [0, 0, 0],
                'force': [0, 0, 0]
            },
            
            {
                'key': 3,
                'name': 'Mars',
                'mass': 6.39e23,
                'radius': 3.389e6,
                'position': [0, 2.279e11, 0],
                'velocity': [-24130, 0, 0],
                'acceleration': [0, 0, 0],
                'force': [0, 0, 0]
            },
            
            {
                'key': 4,
                'name': 'Venus',
                'mass': 4.867e24,
                'radius': 6.051e6,
                'position': [0, 1.082e11, 0],
                'velocity': [-35020, 0, 0],
                'acceleration': [0, 0, 0],
                'force': [0, 0, 0]    
            },
            
            {
                
            'key': 5,
                'name': 'Mercury',
                'mass': 3.285e23,
                'radius': 2.439e6,
                'position': [0, 5.791e10, 0],
                'velocity': [-47360, 0, 0],
                'acceleration': [0, 0, 0],
                'force': [0, 0, 0]
            },
            
            {
                'key': 6,
                'name': 'Jupiter',
                'mass': 1.898e27,
                'radius': 6.991e7,
                'position': [0, 7.785e11, 0],
                'velocity': [-13070, 0, 0],
                'acceleration': [0, 0, 0],
                'force': [0, 0, 0]
            },
            
            {
                'key': 7,
                'name': 'Saturn',
                'mass': 5.683e26,
                'radius': 5.823e7,
                'position': [0, 1.433e12, 0],
                'velocity': [-9690, 0, 0],
                'acceleration': [0, 0, 0],
                'force': [0, 0, 0]
            },
            
            {
                'key': 8,
                'name': 'Uranus',
                'mass': 8.681e25,
                'radius': 2.536e7,
                'position': [0, 2.877e12, 0],
                'velocity': [-6810, 0, 0],
                'acceleration': [0, 0, 0],
                'force': [0, 0, 0]
            },
            
            {
                'key': 9,
                'name': 'Neptune',
                'mass': 1.024e26,
                'radius': 2.462e7,
                'position': [0, 4.503e12, 0],
                'velocity': [-5430, 0, 0],
                'acceleration': [0, 0, 0],
                'force': [0, 0, 0]
            }
            
            ]

for body in body_data:
    body['position'] = compute_initial_position(body['name']) * C.AU
    body['velocity'] = compute_initial_velocity(body['name']) * C.AU

def construct_bodies(body_data=body_data):
    """
    Constructs bodies and sets up their neighbor relationships.

    Parameters:
        node_data (list): A list of dictionaries, each containing data for initializing a Node.
        neighbor_mapping (dict): A dictionary mapping node indices to lists of tuples (neighbor index, contact area).

    Returns:
        A list of constructed Node instances with neighbor relationships set.
    """
    # Construct nodes
    nodes = [Body(**data) for data in body_data]
    return nodes
