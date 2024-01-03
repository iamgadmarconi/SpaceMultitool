"""
A module that contains constructors for nodes.
"""

from constants import Constants as C
from materials import Material as Mat
from materials import Component as Comp
from thermalmodel_v4 import Node

Aluminium = Mat("Aluminium", 237, 897, 0.9, 0.3)
Steel = Mat("Steel", 50, 500, 0.9, 0.3)
GalliumArsenide = Mat("Gallium Arsenide", 40, 340, 0.8, 0.8)
Electronics = Comp("Electronics", 3000, Aluminium, 0.2)

node_data = [
            {
                'key': 1,
                'name': 'Nadir-North',
                'area': C.A_panel,
                'mass': C.m_panel,
                'material': Aluminium,
                'temperature': 293.15,
                'gamma': 30,
                'rb': 'earth',
                'position': [[1.8, 0.0, 2.5], [54.25, 90, 35.75]],
            },
            {
                'key': 2,
                'name': 'Nadir-South',
                'area': C.A_panel,
                'mass': C.m_panel,
                'material': Aluminium,
                'temperature': 293.15,
                'gamma': 30,
                'rb': 'earth',
                'position': [[0.9, 1.5588, 2.5], [73.01, 59.60, 35.75]],
            },
            {
                'key': 3,
                'name': 'South',
                'area': C.A_panel,
                'mass': C.m_panel,
                'material': Aluminium,
                'temperature': 350.15,
                'gamma': 90,
                'rb': 'sun',
                'position': [[-0.9, 1.5588, 2.5], [106.99, 59.60, 35.75]],
            },
            {
                'key': 4,
                'name': 'Zenith-South',
                'area': C.A_panel,
                'mass': C.m_panel,
                'material': Aluminium,
                'temperature': 293.15,
                'gamma': 30,
                'rb': 'earth',
                'position': [[-1.8, 0.0, 2.5], [125.75, 90, 35.75]],

            },
            {
                'key': 5,
                'name': 'Zenith-North',
                'area': C.A_panel,
                'mass': C.m_panel,
                'material': Aluminium,
                'temperature': 293.15,
                'gamma': 30,
                'rb': 'sun',
                'position': [[-0.9, -1.5588, 2.5], [106.99, 120.40, 35.75]],
            },
            {
                'key': 6,
                'name': 'North',
                'area': C.A_panel,
                'mass': C.m_panel,
                'material': Aluminium,
                'temperature': 293.15,
                'gamma': 90,
                'rb': 'earth',
                'position': [[0.9, -1.5588, 2.5], [73.01, 120.40, 35.75]],
            },
            {
                'key': 7,
                'name': 'Velocity',
                'area': C.A_top,
                'mass': C.m_top,
                'material': Aluminium,
                'temperature': 293.15,
                'gamma': 90,
                'rb': 'earth',
                'position': [[0, 0, 5.0], [90, 90, 0]],
            },
            {
                'key': 8,
                'name': 'Negative-Velocity',
                'area': C.A_top,
                'mass': C.m_top,
                'material': Aluminium,
                'temperature': 293.15,
                'gamma': 90,
                'rb': 'sun',
                'position': [[0, 0, 0], [90, 90, 180]],
            },
            {
                'key': 9,
                'name': 'Electronic-Component',
                'area': C.A_compartment,
                'mass': 10,
                'material': Electronics,
                'temperature': 313.15,
                'gamma': 0,
                'rb': 'component',
                'position': [[0, 0, 3.5], [90, 90, 180]],
            },
            {
                'key': 10,
                'name': 'North-Solar-Array',
                'area': C.A_solar,
                'mass': C.m_array,
                'material': Aluminium,
                'temperature': 293.15,
                'gamma': 0,
                'rb': 'sun',
                'position': [[6, 0, 2.5], [54.25, 90, 35.75]],
            },
            {
                'key': 11,
                'name': 'South-Solar-Array',
                'area': C.A_solar,
                'mass': C.m_array,
                'material': Aluminium,
                'temperature': 293.15,
                'gamma': 0,
                'rb': 'sun',
                'position': [[-6, 0, 2.5], [125.75, 90, 35.75]],
            }
]

# Define neighbor relationships
neighbor_mapping = {
    0: [(1, C.A_contact_side), (5, C.A_contact_side), (6, C.A_contact_top), (7, C.A_contact_boom), (8, C.A_contact_support)],
    1: [(2, C.A_contact_side), (6, C.A_contact_top), (7, C.A_contact_top), (8, C.A_contact_support)],
    2: [(3, C.A_contact_side), (6, C.A_contact_top), (7, C.A_contact_top), (10, C.A_contact_boom), (8, C.A_contact_support)],
    3: [(4, C.A_contact_side), (6, C.A_contact_top), (7, C.A_contact_top), (8, C.A_contact_support)],
    4: [(5, C.A_contact_side), (6, C.A_contact_top), (7, C.A_contact_top), (8, C.A_contact_support)],
    5: [(6, C.A_contact_top), (7, C.A_contact_top), (9, C.A_contact_boom), (8, C.A_contact_support)],
    6: [(7, C.A_contact_top), (6, C.A_contact_top)],
    # Continue for other nodes if needed
}


def construct_nodes(node_data=node_data, neighbor_mapping=neighbor_mapping):
    """
    Constructs nodes and sets up their neighbor relationships.

    Parameters:
        node_data (list): A list of dictionaries, each containing data for initializing a Node.
        neighbor_mapping (dict): A dictionary mapping node indices to lists of tuples (neighbor index, contact area).

    Returns:
        A list of constructed Node instances with neighbor relationships set.
    """
    # Construct nodes
    nodes = [Node(**data) for data in node_data]

    # Add neighbors based on mapping
    for node_index, neighbors in neighbor_mapping.items():
        for neighbor_index, contact_area in neighbors:
            nodes[node_index].add_neighbor(nodes[neighbor_index], contact_area)

    return nodes
