import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

class Viewer:
    """
    Class for viewing the trajectory of a body.

    Methods:
        plot: Plots the trajectory of the body.
        plot_interactive: Plots the trajectory of the body interactively with a time slider.
    """
    __slots__ = ['op', 'rs']

    def __init__(self, op: list) -> None:
        """
        Viewer constructor.
        
        parameters:
            op (OrbitPropagator): The orbit propagator to plot.
            rs (np.array): The positions of the body at each time step.
        """
        self.op = op
        if not isinstance(op, list):
            self.op = [op]

        self.rs = [rs for rs in [op.ode_solver()[2] for op in self.op]]

    def plot(self) -> None:
        """
        Plots the trajectory of the body.
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        legend_handles = [Line2D([0], [0], marker='o', color='w', label=f'{obj.name}', markerfacecolor='k', markersize=10) for obj in self.op]

        for i in range(len(self.op)):
            r = self.rs[i]
            obj = self.op[i]
            # Plot each trajectory
            ax.plot(r[:, 0], r[:, 1], r[:, 2], '-', label=f'{self.op[i].name} Trajectory')
            ax.plot(r[0, 0], r[0, 1], r[0, 2], 'o', label=f'{self.op[i].name} Start')

            # Assuming each object has the same central body for simplicity
            r_plot = obj.cb['radius']

            _u, _v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            _x = r_plot * np.cos(_u) * np.sin(_v)
            _y = r_plot * np.sin(_u) * np.sin(_v)
            _z = r_plot * np.cos(_v)

            # Central body surface
            ax.plot_surface(_x + obj.cb['position'][0], _y + obj.cb['position'][1], _z + obj.cb['position'][2], cmap='Blues', alpha=0.3, label=f'{self.op[i].cb["name"]}')

        # Reference lines
        l = r_plot * 2.0
        x, y, z = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        u, v, w = [[l, 0, 0], [0, l, 0], [0, 0, l]]
        ax.quiver(x, y, z, u, v, w, color='k')

        # Setting plot limits and labels
        all_positions = np.array([pos for op in self.op for pos in op.history.values()])
        max_val = np.max(np.abs(all_positions))
        ax.set_xlim([-max_val, max_val])
        ax.set_ylim([-max_val, max_val])
        ax.set_zlim([-max_val, max_val])
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')

        ax.legend(handles=legend_handles)

        plt.show()

    def plot_interactive(self):
        """
        Plots the trajectory of the body interactively with a time slider.
        """
        # Initialize the figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Create legend handles
        legend_handles = [Line2D([0], [0], marker='o', color='w', label=f'{obj.name}', markerfacecolor='k', markersize=10) for obj in self.op]

        # Function to update the plot
        def update_plot(val):
            ax.clear()

            t = int(val)
            for obj in self.op:
                # Plot each central body
                r_plot = obj.cb['radius']
                _u, _v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                _x = r_plot * np.cos(_u) * np.sin(_v)
                _y = r_plot * np.sin(_u) * np.sin(_v)
                _z = r_plot * np.cos(_v)
                ax.plot_surface(_x + obj.cb['position'][0], _y + obj.cb['position'][1], _z + obj.cb['position'][2], cmap='Blues', alpha=0.3)

                # Trace the object's path up to the current time
                times = sorted(obj.history.keys())
                past_positions = np.array([obj.history[time] for time in times if time <= t])
                ax.plot(past_positions[:, 0], past_positions[:, 1], past_positions[:, 2], label=f'{obj.name} Path')

                # Plot the object's position at the current time
                pos = obj.history[t]
                ax.scatter(pos[0], pos[1], pos[2])

            # Set plot limits and labels
            all_positions = np.array([pos for op in self.op for pos in op.history.values()])
            max_val = np.max(np.abs(all_positions))
            ax.set_xlim([-max_val, max_val])
            ax.set_ylim([-max_val, max_val])
            ax.set_zlim([-max_val, max_val])
            ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
            plt.legend(handles=legend_handles)

            plt.draw()

        # Slider
        ax_slider = plt.axes([0.1, 0.02, 0.8, 0.03])
        slider = Slider(ax_slider, 'Time', 0, np.max([op.tspan for op in self.op]), valinit=0, valstep=np.min([op.dt for op in self.op]))

        # Update the plot when the slider is changed
        slider.on_changed(update_plot)

        # Initial plot
        update_plot(0)

        plt.show()


    def plot_animated(self, interval=200):
        """
        Creates an animation of the space object's trajectory.

        parameters:
            interval (int): The interval between each frame in milliseconds (default is 200).
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        legend_handles = [Line2D([0], [0], marker='o', color='w', label=f'{obj.name}', markerfacecolor='k', markersize=10) for obj in self.op]

        # Prepare data for the animation and initialize plots
        plots = []
        for obj in self.op:
            obj_plot, = ax.plot([], [], [], 'o', label=f'{obj.name} Position')
            trail_plot, = ax.plot([], [], [], '-', label=f'{obj.name} Trajectory')
            plots.append((obj_plot, trail_plot))

        # Initialize the plot structure
        def init():
            for obj_plot, trail_plot in plots:
                obj_plot.set_data([], [])
                obj_plot.set_3d_properties([])
                trail_plot.set_data([], [])
                trail_plot.set_3d_properties([])
            return [plot for subplots in plots for plot in subplots]

        # Update function for each frame
        def update(frame):
            for obj, (obj_plot, trail_plot) in zip(self.op, plots):


                r_plot = obj.cb['radius']

                _u, _v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                _x = r_plot * np.cos(_u) * np.sin(_v)
                _y = r_plot * np.sin(_u) * np.sin(_v)
                _z = r_plot * np.cos(_v)

                ax.plot_surface(_x + obj.cb['position'][0], _y + obj.cb['position'][1], _z + obj.cb['position'][2], cmap='Blues', alpha=0.3, label=f'{obj.cb["name"]}')

                l = r_plot*2.0

                x,y,z = [[0,0,0], [0,0,0], [0,0,0]]
                u,v,w = [[l,0,0], [0,l,0], [0,0,l]]
                ax.quiver(x,y,z,u,v,w, color='k')

                times = sorted(obj.history.keys())
                t = times[min(frame, len(times) - 1)]
                pos = obj.history[t]

                # Update the position of the object
                obj_plot.set_data(pos[0], pos[1])
                obj_plot.set_3d_properties(pos[2])

                # Update the trail
                trail_xs, trail_ys, trail_zs = zip(*[obj.history[time] for time in times[:frame+1]])
                trail_plot.set_data(trail_xs, trail_ys)
                trail_plot.set_3d_properties(trail_zs)

            return [plot for subplots in plots for plot in subplots]

        # Setting plot limits and labels
        all_positions = np.array([pos for op in self.op for pos in op.history.values()])
        max_val = np.max(np.abs(all_positions))
        ax.set_xlim([-max_val, max_val])
        ax.set_ylim([-max_val, max_val])
        ax.set_zlim([-max_val, max_val])
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        plt.legend(handles=legend_handles)

        # Creating the animation
        ani = FuncAnimation(fig, update, frames=np.max([op.tspan for op in self.op]), init_func=init, interval=interval, blit=True)

        plt.show()