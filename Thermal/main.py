"""
Main file for the thermal model.
"""

import numpy as np
from thermalmodel_v4 import ThermalModel as TM4
from viewer import plot_3d
from nodes import construct_nodes


def main():
    """
    Main function.
    """
    nodes = construct_nodes()
    tm = TM4(nodes)
    beta_range = np.linspace(0, 90, 2)
    h_range = np.linspace(200, 2000, 2)
    time_range = np.linspace(0, 31000, 1000)

    plot_3d(tm, beta_range, h_range, time_range)


if __name__ == "__main__":
    main()
