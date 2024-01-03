import numpy as np
from scipy.integrate import ode


class OrbitPropagator:
    """
    Propagates an orbit using the given initial conditions and time span.
    """
    __slots__ = ['r0', 'v0', 'tspan', 'dt', 'cb', 'history', 'name']

    def __init__(self, r0: list, v0: list, tspan: float, dt: float, cb: dict, name="Object") -> None:
        """
        OrbitPropagator constructor.
        
        Parameters:
            r0 (np.array): The initial position.
            v0 (no.array): The initial velocity.
            tspan (float): The time span to propagate the orbit for.
            dt (float): The time step to use.
            cb (dict): A dictionary containing the celestial body's properties.
        """
        self.r0 = np.array(r0)
        self.v0 = np.array(v0)
        self.tspan = tspan
        self.dt = dt
        self.cb = cb
        self.history = {0: self.r0.copy()}
        self.name = name

    def __str__(self) -> str:
        return f'{self.name} orbiting {self.cb["name"]}'

    def __repr__(self) -> str:
        return self.__str__()

    def ode_solver(self) -> tuple:
        """
        Solves the ODE for the given time and initial conditions.

        Parameters:
            time (float): The time to solve the ODE for.
            initial_conditions (list): The initial conditions.
            mu (float): The gravitational parameter of the body.
            dt (float): The time step to use.
        Returns:
            tuple: The solution to the ODE.
        """
        r0, v0 = self.r0, self.v0
        y0 = np.concatenate((r0, v0))
        def derivatives(t, y, mu):
            rx, ry, rz, vx, vy, vz = y
            r = np.array([rx, ry, rz])
            norm_r = np.linalg.norm(r)
            ax, ay, az = -mu * r / norm_r ** 3 
            return [vx, vy, vz, ax, ay, az]

        solver = ode(derivatives).set_integrator('dop853')  # Changed to dop853 for better performance
        solver.set_initial_value(y0, 0).set_f_params(self.cb['mu'])

        n_steps = int(np.ceil(self.tspan / self.dt))
        ys = np.zeros((n_steps, 6))
        ys[0] = np.array(y0)
        ts = np.zeros(n_steps)

        step = 0

        while solver.successful() and step < n_steps:
            solver.integrate(solver.t + self.dt)
            ts[step] = solver.t
            ys[step] = solver.y
            step += 1
            self.history[step * self.dt] = solver.y[:3].copy()

        return ys, ts, ys[:,:3]
    
