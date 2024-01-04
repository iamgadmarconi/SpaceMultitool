import numpy as np
from scipy.integrate import ode
from constants.math_utils import coes2rv, rv2coes
from constants.data_handling import perturbations, stop_conditions, check_deorbit, check_max_alt, check_min_alt

class OrbitPropagator:
    """
    Propagates an orbit using the given initial conditions and time span.
    """
    __slots__ = ['state', 'r0', 'v0', 'tspan', 'dt', 'cb', 'history', 'name', 'coes0', 'COES', 'flags', 'sc']

    def __init__(self, state: list, tspan: float, dt: float, cb: dict, name="Object", COES=False, perts=perturbations(), sc=stop_conditions()) -> None:
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
            perturbations (dict): A dictionary containing the perturbations to use.
            stop_conditions (dict): A dictionary containing the stop conditions to use.
        """
        if COES:
            self.r0, self.v0 = coes2rv(state, cb['mu'])
            self.coes0 = state
        else:
            self.r0 = np.array(state[:3])
            self.v0 = np.array(state[3:])
            self.coes0 = rv2coes(state, cb['mu'])
        
        self.tspan = tspan
        self.dt = dt
        self.cb = cb
        self.history = {0: {'r': self.r0.copy(),
                            'v': self.v0.copy()}}
        self.name = name
        self.flags = perts
        self.sc = sc

        self.sc_map = {'deorbit': check_deorbit,
                       'max_alt': check_max_alt,
                       'min_alt': check_min_alt}
        
        self.sc_func = []
        for key in self.sc.keys():
            self.sc_func.append(self.sc_map[key])

    def __str__(self) -> str:
        return f'{self.name} orbiting {self.cb["name"]}'

    def __repr__(self) -> str:
        return self.__str__()
    
    @staticmethod
    def perturbations(y, mu, flags, **kwargs):
        """
        Calculates the perturbations to the orbit.

        Parameters:
            y (list): The initial conditions.
            mu (float): The gravitational parameter of the body.
            perturbations (dict): A dictionary containing the perturbations to use.
            **kwargs: Additional keyword arguments.
                J2 (float): The J2 perturbation coefficient.
                radius (float): The radius of the body.
                J2,2 (float): The J2,2 perturbation coefficient.
                solar_flux (float): The solar flux.
                drag_coefficient (float): The drag coefficient.
                area_mass_ratio (float): The area to mass ratio.
                lunar_gravity (float): The lunar gravity coefficient.
                lunar_radius (float): The radius of the moon.
                lunar_position (np.array): The position of the moon.
                relativity (float): The relativity coefficient.
        Returns:
            np.array: The perturbations.
        """
        rx, ry, rz, vx, vy, vz = y
        r = np.array([rx, ry, rz])
        v = np.array([vx, vy, vz])
        norm_r = np.linalg.norm(r)

        a_perturb = np.zeros(3)

        if flags['J2']:
            R = kwargs.get('radius', None)
            J2 = kwargs.get('J2', None)
            if R is not None and J2 is not None:
                # J2 perturbation calculation
                z2 = r[2] ** 2
                r2 = norm_r ** 2
                tx = r[0] / norm_r * (5 * z2 / r2 - 1)
                ty = r[1] / norm_r * (5 * z2 / r2 - 1)
                tz = r[2] / norm_r * (5 * z2 / r2 - 3)
                a_perturb += 1.5 * J2 * (R ** 2 / norm_r ** 4) * mu / r2 ** 2 * np.array([tx, ty, tz])
                # factor = 1.5 * J2 * (R / norm_r)**2 * mu / norm_r**4
                # z2_over_r2 = (rz / norm_r)**2
                # a_perturb += factor * ((5 * z2_over_r2 - 1) * r - 2 * rz * np.array([0, 0, 1]))
            else:
                if R is None:
                    raise ValueError('Missing radius')
                elif J2 is None:
                    raise ValueError('Missing J2 perturbation coefficient')

        if flags['J2,2']:
            # Add J2,2 perturbation calculation here
            pass

        if flags['drag']:
            R = kwargs.get('radius', None)
            area = kwargs.get('area', None)
            cd = kwargs.get('cd', None)
            m = kwargs.get('m', None)
            if cd is None:
                cd = 2.2
            if area is not None and R is not None and m is not None:
                z = norm_r - R
                rho = 1.225 * np.exp(-z / 7500)
                v_rel = v - np.cross(np.array([0, 0, 7.292115e-5]), r)
                drag = - v_rel * np.linalg.norm(v_rel) * rho * cd * area / 2 / m

                a_perturb += drag
            else:
                if R is None:
                    raise ValueError('Missing radius')
                elif area is None:
                    raise ValueError('Missing area')
                elif m is None:
                    raise ValueError('Missing mass')
                
        if flags['lunar']:
            # Add lunar perturbation calculation here
            pass

        if flags['srp']:
            # Add solar radiation pressure perturbation calculation here
            pass

        if flags['relativity']:
            # Add relativity perturbation calculation here
            pass

        return a_perturb

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
            """
            Derivatives of the state vector.

            Parameters:
                t (float): The time.
                y (list): The state vector.
                mu (float): The gravitational parameter of the body.
            Returns:
                list: The derivatives of the state vector.
            """
            rx, ry, rz, vx, vy, vz = y
            r = np.array([rx, ry, rz])
            norm_r = np.linalg.norm(r)
            a_gravity = -mu * r / norm_r ** 3 
            a_perturb = self.perturbations(y, mu, self.flags, radius=self.cb['radius'], J2=self.cb['J2'], area=1000, cd=22, m=420000)
            a_total = a_gravity + a_perturb
            return [vx, vy, vz, *a_total]

        solver = ode(derivatives).set_integrator('dop853')  # Changed to dop853 for better performance
        solver.set_initial_value(y0, 0).set_f_params(self.cb['mu'])

        n_steps = int(np.ceil(self.tspan / self.dt))
        ys = np.zeros((n_steps, 6))
        ys[0] = np.array(y0)
        ts = np.zeros(n_steps)

        step = 0

        #with tqdm(total=n_steps, desc=f"{self.name} ODE Integration Progress") as pbar:
        for step in range(n_steps):
            if not solver.successful() and not self.stop_conditions():
                break
            solver.integrate(solver.t + self.dt)
            ts[step] = solver.t
            ys[step] = solver.y
            self.history[step * self.dt] = {'r': solver.y[:3].copy(),
                                            'v': solver.y[3:].copy()}
            # pbar.update(1)  # Update progress after each step

        return ys, ts, ys[:,:3]
    
