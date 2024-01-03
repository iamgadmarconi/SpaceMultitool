"""
A module that contains various functions for plotting the results of the thermal calculations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def plot_3d(tm, beta_range, h_range, time_range):
    """
    Plots a 3D graph of temperature variation over time and beta angle for different altitudes.

    
    Parameters:
        tm (ThermalModel): The thermal model object.
        beta_range (numpy.ndarray): The range of beta angles to plot.
        h_range (numpy.ndarray): The range of altitudes to plot.
        time_range (numpy.ndarray): The range of times to plot.
    """

    # Number of nodes
    temperatures = tm.integrate_heat_balance(beta_range, h_range, time_range)
    node_keys = list(tm.nodes.keys())  # Get the keys (identifiers) of the nodes
    num_nodes = len(node_keys)
    num_cols = 5
    num_rows = num_nodes // num_cols + (num_nodes % num_cols > 0)
    fig, axs = plt.subplots(num_rows, num_cols, subplot_kw={'projection': '3d'}, figsize=(20, num_rows * 4))
    axs = axs.ravel() if num_rows > 1 else [axs]
    global_min_temp = np.min(temperatures)
    global_max_temp = np.max(temperatures)

    def update_plot(val):
        """
        Update the plots based on the slider value.
        """
        h_index = np.argmin(np.abs(h_range - val))  # Find closest index in h_range

        for i, ax in enumerate(axs):
            if i < num_nodes:
                node_key = node_keys[i]
                ax.clear()
                X, Y = np.meshgrid(time_range, beta_range)
                Z = temperatures[:, h_index, :, i]
                ax.plot_surface(X, Y, Z, cmap='viridis', vmin=global_min_temp, vmax=global_max_temp)
                ax.set_title(tm.nodes[node_key].name)  # Set title using node name
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Beta Angle (deg)')
                ax.set_zlabel('Temperature (K)')
                ax.set_zlim(global_min_temp, global_max_temp)
            else:
                ax.axis('off')

        fig.canvas.draw_idle()

    plt.subplots_adjust(bottom=0.25)
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03])
    slider = Slider(ax_slider, 'Altitude (h)', h_range[0], h_range[-1], valinit=h_range[0])
    slider.on_changed(update_plot)
    update_plot(h_range[0])
    plt.show()
