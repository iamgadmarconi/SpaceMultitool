import numpy as np
from scipy.integrate import ode
from constants.math_utils import coes2rv


class OrbitPropagator:
    """
    Propagates an orbit using the given initial conditions and time span.
    """
    __slots__ = ['state', 'r0', 'v0', 'tspan', 'dt', 'cb', 'history', 'name', 'COES']

    def __init__(self, state: list, tspan: float, dt: float, cb: dict, name="Object", COES=False) -> None:
        """
        OrbitPropagator constructor.
        
        Parameters:
            state (list): The initial conditions of the orbit.
                if COES is False: [r0, v0]
                if COES is True: [a, e, i, RAAN, argp, nu]
            tspan (float): The time span to propagate the orbit for.
            dt (float): The time step to use.
            cb (dict): A dictionary containing the celestial body's properties.
            name (str): The name of the object.
            COES (bool): Whether the state is in COES or not. Defaults to False.
                Semi-major axis (a), eccentricity (e), inclination (i), right ascension of the ascending node (raan), argument of perigee (argp), true anomaly (nu)
        """
        if COES:
            self.r0, self.v0 = coes2rv(state, cb['mu'])
        else:
            self.r0 = np.array(state[:3])
            self.v0 = np.array(state[3:])

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
    
