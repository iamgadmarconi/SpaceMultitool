import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, Manager
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import time
from tqdm import tqdm
from constants.math_utils import rv2coes
from physics.engine import OrbitPropagator
from models.celestial_body import CelestialBody


class Viewer:
    """
    Class for viewing the trajectory of a body.

    Methods:
        plot: Plots the trajectory of the body.
        plot_interactive: Plots the trajectory of the body interactively with a time slider.
    """
    __slots__ = ['op', 'rs', 'mode']

    def __init__(self, op: list=[], mode: str=None, mp: bool=False) -> None:
        """
        Viewer constructor.
        
        Parameters:
            op (OrbitPropagator): The orbit propagator to plot.
            rs (np.array): The positions of the body at each time step.
            mode (str): The mode to use for plotting. Can be 'static', 'interactive', or 'animated'.
            mp (bool): Whether to use multiprocessing or not.
        """
        self.op = op
        if not isinstance(op, list):
            self.op = [op]

        if not mp:
            self.rs = [rs for rs in [op.ode_solver()[2] for op in self.op]]
        
        elif mp:
            with Pool() as pool, Manager() as manager:
                indices = manager.list(range(len(self.op)))
                results = pool.starmap(self.ode_solver_worker, [(op, idx, len(self.op)) for idx, op in enumerate(self.op)])

            self.rs = list(results)

        if mode is None:
            pass
        elif mode == 'static':
            self.plot_static()
        elif mode == 'interactive':
            self.plot_interactive()
        elif mode == 'animated':
            self.plot_animated()
        else:
            raise ValueError(f'Invalid mode: {mode}')

    @staticmethod
    def ode_solver_worker(op: OrbitPropagator, idx: int, total_ops: int) -> np.array:
        """
        Solves the ode for a single orbit propagator.

        Parameters:
            op (OrbitPropagator): The orbit propagator to solve.
            idx (int): The index of the orbit propagator.
            total_ops (int): The total number of orbit propagators.
        Returns:
            np.array: The positions of the body at each time step.
        """
        with tqdm(total=100, desc=f"Process {idx}", position=idx, leave=False) as pbar:
            # Replace the following line with actual progress updates
            for _ in range(100):
                pbar.update(1)
                # Simulate work being done
                time.sleep(0.1)
        # Return the result of ode_solver
        return op.ode_solver()[2]
    
    def plot_n_orbits(self, rs, radius=6378000) -> None:
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(111, projection='3d')

        n = 0

        for n, r in enumerate(rs):
            # Flatten the first dimension of r
            r = np.squeeze(r)

            # Now you can plot r
            ax.plot(r[:, 0], r[:, 1], r[:, 2], '-', label=f'Orbit {n}')
            ax.plot(r[0, 0], r[0, 1], r[0, 2], 'o', label=f'Orbit {n} Start')
            n += 1

        r_plot = radius
        _u, _v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        _x = r_plot * np.cos(_u) * np.sin(_v)
        _y = r_plot * np.sin(_u) * np.sin(_v)
        _z = r_plot * np.cos(_v)
        ax.plot_surface(_x, _y, _z, cmap='Blues', alpha=0.3)

        l = r_plot * 2.0
        x, y, z = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        u, v, w = [[l, 0, 0], [0, l, 0], [0, 0, l]]
        ax.quiver(x, y, z, u, v, w, color='k')

        max_val = np.max(np.abs(rs))

        ax.set_xlim([-max_val, max_val])
        ax.set_ylim([-max_val, max_val])
        ax.set_zlim([-max_val, max_val])

        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')

        ax.set_title(f'Trajectory')

        plt.show()

    def plot_static(self) -> None:
        """
        Plots the trajectory of the body.
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        legend_handles = []

        for i in range(len(self.op)):
            r = self.rs[i]
            obj = self.op[i]
            print(r, obj)
            # Plot each trajectory
            line, = ax.plot(r[:, 0], r[:, 1], r[:, 2], '-', label=f'{obj.name} Trajectory')
            ax.plot(r[0, 0], r[0, 1], r[0, 2], 'o', label=f'{obj.name} Start')

            # Create a legend entry with the same color as the line
            legend_handles.append(Line2D([0], [0], color=line.get_color(), label=f'{obj.name} Trajectory'))

            # Assuming each object has the same central body for simplicity
            r_plot = obj.cb['radius']

            _u, _v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            _x = r_plot * np.cos(_u) * np.sin(_v)
            _y = r_plot * np.sin(_u) * np.sin(_v)
            _z = r_plot * np.cos(_v)

            # Central body surface
            if isinstance(obj, OrbitPropagator):
                ax.plot_surface(_x + obj.cb['position'][0], _y + obj.cb['position'][1], _z + obj.cb['position'][2], cmap='Blues', alpha=0.3, label=f'{self.op[i].cb["name"]}')

        # Reference lines
        l = r_plot * 2.0
        x, y, z = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        u, v, w = [[l, 0, 0], [0, l, 0], [0, 0, l]]
        ax.quiver(x, y, z, u, v, w, color='k')

        # Setting plot limits and labels
        if isinstance(obj, OrbitPropagator):
            all_positions = np.array([pos['r'] for op in self.op for pos in op.history.values()])
        elif isinstance(obj, CelestialBody):
            all_positions = np.array([r for op in self.op for r in op.rs_list])

        max_val = np.max(np.abs(all_positions))
        ax.set_xlim([-max_val, max_val])
        ax.set_ylim([-max_val, max_val])
        ax.set_zlim([-max_val, max_val])
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')

        ax.legend(handles=legend_handles, loc='upper left')

        plt.show()

    def plot_interactive(self):
        """
        Plots the trajectory of the body interactively with a time slider.
        """
        # Initialize the figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

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
                past_positions = np.array([obj.history[time]['r'] for time in times if time <= t])
                line, = ax.plot(past_positions[:, 0], past_positions[:, 1], past_positions[:, 2], label=f'{obj.name} Path')

                # Plot the object's position at the current time
                pos = obj.history[t]['r']
                ax.scatter(pos[0], pos[1], pos[2])

            # Set plot limits and labels
            all_positions = np.array([pos['r'] for op in self.op for pos in op.history.values()])
            max_val = np.max(np.abs(all_positions))
            ax.set_xlim([-max_val, max_val])
            ax.set_ylim([-max_val, max_val])
            ax.set_zlim([-max_val, max_val])
            ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')

            # Update the legend
            ax.legend(loc='upper left')

            plt.draw()

        # Slider
        ax_slider = plt.axes([0.1, 0.02, 0.8, 0.03])
        slider = Slider(ax_slider, 'Time', 0, np.max([op.tspan for op in self.op]), valinit=0, valstep=np.min([op.dt for op in self.op]))

        # Update the plot when the slider is changed
        slider.on_changed(update_plot)

        # Initial plot
        update_plot(0)

        plt.show()

    def plot_animated(self, interval=10):
        """
        Creates an animation of the space object's trajectory.

        Parameters:
            interval (int): The interval between each frame in milliseconds (default is 200).
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

        legend_handles = []

        # Prepare data for the animation and initialize plots
        plots = []
        for obj in self.op:
            obj_plot, = ax.plot([], [], [], 'o', label=f'{obj.name} Position')
            trail_plot, = ax.plot([], [], [], '-', label=f'{obj.name} Trajectory')
            plots.append((obj_plot, trail_plot))
            plot_color = trail_plot.get_color()
            legend_handles.append(Line2D([0], [0], color=plot_color, label=f'{obj.name}', markerfacecolor=plot_color, markersize=10))

        central_body_ids = set(obj.cb['name'] for obj in self.op)
        central_bodies = {cb_id: next(obj.cb for obj in self.op if obj.cb['name'] == cb_id) for cb_id in central_body_ids}

        for cb_id, cb in central_bodies.items():
            # Plot each central body
            r_plot = cb['radius']
            _u, _v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            _x = r_plot * np.cos(_u) * np.sin(_v)
            _y = r_plot * np.sin(_u) * np.sin(_v)
            _z = r_plot * np.cos(_v)
            ax.plot_surface(_x + cb['position'][0], _y + cb['position'][1], _z + cb['position'][2], cmap='Blues', alpha=0.3)

        # Initialize the plot structure
        def init():
            """
            Initialize the plot.

            Returns:
                list: The plots.
            """
            for obj_plot, trail_plot in plots:
                obj_plot.set_data([], [])
                obj_plot.set_3d_properties([])
                trail_plot.set_data([], [])
                trail_plot.set_3d_properties([])

            return [plot for subplots in plots for plot in subplots]

        def update(frame):
            """
            Update the plot for each frame.
            Parameters:
                frame (int): The current frame.
            Returns:
                list: The updated plots.
            """
            for obj, (obj_plot, trail_plot) in zip(self.op, plots):
                times = sorted(obj.history.keys())
                if frame < len(times):
                    t = times[frame]
                    pos = obj.history[t]['r']
                    # Update object position
                    obj_plot.set_data([pos[0]], [pos[1]])
                    obj_plot.set_3d_properties([pos[2]])

                    # Update object trail
                    trail_xs, trail_ys, trail_zs = zip(*[obj.history[time]['r'] for time in times[:frame+1]])
                    trail_plot.set_data(trail_xs, trail_ys)
                    trail_plot.set_3d_properties(trail_zs)

            return [plot for subplots in plots for plot in subplots]

        # Setting plot limits and labels
        all_positions = np.array([pos['r'] for op in self.op for pos in op.history.values()])
        max_val = np.max(np.abs(all_positions))
        ax.set_xlim([-max_val, max_val])
        ax.set_ylim([-max_val, max_val])
        ax.set_zlim([-max_val, max_val])
        ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
        plt.legend(handles=legend_handles, loc='upper left')

        # Creating the animation
        ani = FuncAnimation(fig, update, frames=np.max([op.tspan for op in self.op]), init_func=init, interval=interval, blit=False)

        plt.show()

    def plot_coes(self, op_index=0):
        """
        Plots the COEs of the body over time.

        Parameters:
            op_index (int): The index of the orbit propagator to plot.
        """
        times = np.arange(0, self.op[op_index].tspan, self.op[op_index].dt)
        coes = []  # To store COEs at each time step

        for time in times:
            r = self.op[op_index].history[time]['r']
            v = self.op[op_index].history[time]['v']
            coes_at_t = rv2coes(r, v, self.op[op_index].cb['mu'])
            coes.append(coes_at_t)

        coes = np.array(coes)  # Convert to numpy array for easier manipulation
        plt.figure(figsize=(10, 10))

        coes_name = ['Semi-major Axis (m)', 'Eccentricity', 'Inclination (deg)', 'RAAN (deg)', 'Argument of Perigee (deg)', 'True Anomaly (deg)']
        for i in range(coes.shape[1]):
            plt.subplot(coes.shape[1], 1, i + 1)
            plt.plot(times, coes[:, i])
            plt.ylabel(coes_name[i])
            plt.xlabel('Time (s)')

        plt.show()