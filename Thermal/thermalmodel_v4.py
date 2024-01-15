"""
A module for the thermal model.
"""

from multiprocessing import Pool
import numpy as np
from tqdm import tqdm
from scipy.integrate import solve_ivp, dblquad
from constants import Constants as C
from materials import Component, Material


class Node:
    """
    Represents a node in the thermal model.

    Methods:
        name(name): Returns the name of the node.
        add_neighbor(node, contact_area, contact_conductance): Adds a neighbor to the node.
        remove_neighbor(key): Removes a neighbor from the node.
        get_neighbors(): Returns a list of neighbors.
        update_temperature(temperature): Updates the temperature of the node.
        get_effective_emissivity(param): Calculate the effective emissivity of the node with MLI.
        calculate_flux(index): Calculate the radiation flux terms F1 and F2 for a given layer in the MLI.
    """

    __slots__ = ['key', 'neighbors', 'area', 'conductivity', '_mass', '_cp', 'emissivity', 'absorptivity', 'temperature', 'gamma',
                 'radiating_body', 'position', 'mli', 'heat_flux_int', 'name', '_thermal_mass']

    def __init__(self, key: int, name: str, area: float, mass: float, material: Material, temperature: float, gamma: float, rb: str, position: list, mli: list=None) -> None:
        """ 
        Initializes a new instance of the Node class.

        Parameters:
            key (int): Unique identifier for the node.
            name (str): Name of the node.
            area (float): Incident area. m^2
            conductivity (float): Thermal conductivity. W/mK
            mass (float): Mass of the node. kg
            cp (float): Specific heat capacity. J/kgK
            emissivity (float): Emissivity.
            absorptivity (float): Absorptivity. 
            temperature (float): Temperature. K
            gamma (float): Angle between the node and the sun/earth. degrees
            rb (str): Radiating body (earth or sun).
            position (list): Position of the center of the node from the geometric center of the S/C in meters 
                             and angle between the normal vector of the node face and the origin in degrees 
                             ([x, y, z], [theta_xy, theta_yz, theta_xz]).
            mli (bool): True if the node has MLI, False otherwise.
            heat_flux (float): Heat flux in watts (default: 0).
            mli (dict): Additional arguments for the MLI model.
                [{temperature: float, emissivity: float, absorptivity: float, cp: float, area: float, mass: float, xi: float]}]

        """
        self.key = key
        self.name = name
        self.neighbors = {}
        self.area = area
        self.conductivity = material.conductivity
        self._mass = mass
        self._cp = material.cp
        self.emissivity = material.emissivity
        self.absorptivity = material.absorptivity
        self.temperature = temperature
        self.gamma = gamma
        self.radiating_body = rb
        self.position = np.array(position)
        self.heat_flux_int = 0
        self.mli = mli
        if self.mli is not None:
            self.emissivity = self.get_emissivity() # Calculate the effective emissivity of the node with MLI
        if isinstance(material, Component):
            self.heat_flux_int = material.power * material.efficiency # if node is electronic component, heat flux in W
        self._thermal_mass = self._mass * self._cp

    @property
    def thermal_mass(self):
        """
        Returns the thermal mass of the node.

        Returns:
            float: Thermal mass of the node.
        """
        return self._thermal_mass

    def change_name(self, name):
        """
        Changes the name of the node.

        Parameters:
            name (str): Name of the node.

        Returns:
            str: The name of the node.
        """
        self.name = name
        return self.name

    def __str__(self) -> str:
        """
        Returns a string representation of the node.

        Returns:
            str: A string representation of the node.
        """
        return f"{self.name}"

    def __repr__(self) -> str:
        """
        Returns a string representation of the node.

        Returns:
            str: A string representation of the node.
        """
        neighbor_ids = list(self.neighbors.keys())
        return f"Node {self.key} with neighbors: {', '.join(map(str, neighbor_ids))}"

    def add_neighbor(self, node, contact_area):
        """
        Adds a neighbor to the node.

        Parameters:
            node (Node): The neighbor node to be added.
            contact_area (float): Contact area between the node and the neighbor. m^2
        """
        # check if neighbor already exists
        if node.key not in self.neighbors:
            self.neighbors[node.key] = (node, contact_area)

        # Add this node as a neighbor to the added neighbor, if not already present
        if self.key not in node.neighbors:
            node.neighbors[self.key] = (self, contact_area)
   
    def remove_neighbor(self, key):
        """
        Removes a neighbor from the node.

        Parameters:
            key (int): The key of the neighbor node to be removed.
        """
        if key in self.neighbors:
            del self.neighbors[key]

    def get_neighbors(self):
        """
        Returns an iterator over the set of neighbors.

        Returns:
            iterator: An iterator over the set of neighbors.
        """
        return iter(self.neighbors.values())

    def update_temperature(self, temperature):
        """
        Updates the temperature of the node.

        Parameters:
            temperature (float): The new temperature value.
        """
        # if self.heat_flux_int == 0:  # Update temperature only if it's not an electronic component
        self.temperature = temperature
        
    def get_emissivity(self):
        """
        Calculate the effective emissivity of the node with MLI, using the provided physical model.

        Returns:
            float: Effective emissivity of the node.
        """
        total_radiative_power = 0

        # Iterate through each layer to calculate the radiative power
        for i, layer in enumerate(self.mli):
            F1, F2 = self.calculate_flux(i)
            # Total power radiated by this layer
            total_radiative_power += layer['xi'] * layer['emissivity'] * (F1 + F2 - 2 * C.sigma * self.temperature**4)

        # The effective emissivity is the total radiative power divided by the Stefan-Boltzmann law for the entire node
        effective_emissivity = total_radiative_power / (C.sigma * self.area * self.temperature**4)

        # Ensure emissivity is within physical bounds [0, 1]
        effective_emissivity = max(0, min(effective_emissivity, 1))

        return effective_emissivity

    def calculate_flux(self, index):
        """
        Calculate the radiation flux terms F1 and F2 for a given layer in the MLI.

        Parameters:
            index (int): The index of the layer within the MLI stack.

        Returns:
            tuple: A tuple containing the F1 and F2 flux terms for the given layer.
        """
        # Retrieve the properties of the current layer and adjacent layers
        current_layer = self.mli[index]
        epsilon_i = current_layer['emissivity']
        alpha_i = current_layer['absorptivity']
        T_i = current_layer['temperature']  # Temperature of the current layer

        # Calculate the radiative conductance for the current layer
        k_rad_i = C.sigma * (epsilon_i + alpha_i) / (1 - epsilon_i)
        
        # Determine the temperature of the adjacent layers
        # For the outermost layers, if there's no adjacent MLI, use the node's temperature
        T_above = self.mli[index - 1]['temperature'] if index > 0 else self.temperature
        T_below = self.mli[index + 1]['temperature'] if index < len(self.mli) - 1 else self.temperature

        # Calculate the radiation flux terms F1 and F2
        F1 = k_rad_i * (T_above**4 - T_i**4)  # Flux from the layer above to the current layer
        F2 = k_rad_i * (T_i**4 - T_below**4)  # Flux from the current layer to the layer below

        return F1, F2


class OrbitProperties():
    """
    Class for orbit properties.

    Attributes:
        altitude (float): Altitude in km.
        beta (float): Angle between orbit and equator in degrees.
        t (float): Time in seconds.
    """

    __slots__ = ['_h', '_beta', '_period_cache', '_beta_critical_cache', '_albedo', '_earth_ir']

    def __init__(self, altitude, beta) -> None:
        """
        Initialize the OrbitProperties class.

        Parameters:
            altitude (float): Altitude in km.
            beta (float): Angle between orbit and equator in degrees.
            t (float): Time in seconds.
        """
        self._h = altitude
        self._beta = beta
        self._period_cache = {}
        self._beta_critical_cache = {}
        self._albedo = self.albedo()
        self._earth_ir = self.earth_ir()

    @property
    def h(self):
        """
        Gets the altitude.

        Returns:
            float: Altitude in km.
        """
        return self._h

    @h.setter
    def h(self, value):
        """
        Sets the altitude and invalidates the cached period value.

        Parameters:
            value (float): Altitude in km.
        """
        self._h = value
        self._period_cache = {}
        self._beta_critical_cache = {}

    @property
    def beta(self):
        """
        Gets the beta angle.

        Returns:
            float: Beta angle in degrees.
        """
        return self._beta

    @beta.setter
    def beta(self, value):
        """
        Sets the beta angle and invalidates the cached values that depend on it.

        Parameters:
            value (float): Beta angle in degrees.
        """
        self._beta = value
        self._beta_critical_cache = None  # Invalidate the cached beta critical value

    def period(self):
        """
        Calculate the orbital period in seconds.

        Returns:
            float: Orbital period in seconds.
        """

        if self._h not in self._period_cache:
            a = (C.r_earth + self.h) * 1000 # Convert altitude to meters
            self._period_cache[self._h] = 2 * np.pi * np.sqrt(a**3 / C.mu)
        return self._period_cache[self._h]

    def beta_critical(self):
        """
        Calculate the angle at which eclipse will be a factor to consider.

        Returns:
            float: Angle in radians.
        """
        if self._h not in self._beta_critical_cache:
            self._beta_critical_cache[self._h] = np.arcsin(C.r_earth / (C.r_earth + self._h))
        return self._beta_critical_cache[self._h]

    def eclipse_fraction(self):
        """
        Calculate the fraction of orbit in eclipse.

        Returns:
            float: Fraction of orbit in eclipse.
        """
        if self._h not in self._beta_critical_cache:
            self.beta_critical()  # Ensure that beta_critical is calculated

        beta_rad = np.deg2rad(self.beta)
        if np.abs(beta_rad) < self._beta_critical_cache[self._h] :
            term = np.sqrt((self.h**2 + 2 * C.r_earth * self.h)) / ((C.r_earth + self.h) * np.cos(beta_rad))
            f_e = 1 / np.pi * np.arccos(term)
        else:
            f_e = 0
        return f_e

    def eclipse(self, t):
        """
        Check if the satellite is in eclipse.

        Parameters:
            t (float): Time in seconds.

        Returns:
            bool: True if in eclipse, False if not.
        """
        if self._h not in self._period_cache:
            self.period()  # Ensure that period is calculated

        t_eclipse = np.mod(t, self._period_cache[self._h])
        eclipse_start = self._period_cache[self._h]  / 2 * (1 - self.eclipse_fraction())
        eclipse_end = self._period_cache[self._h]  / 2 * (1 + self.eclipse_fraction())
        return eclipse_start < t_eclipse < eclipse_end

    def view_factor(self, gammas, r_values):
        """
        Calculate the view factor between nodes and earth/sun.

        Parameters:
            gammas (dict: float): Angle between nodes and earth/sun in degrees.
            r_values (dict: str): Irradiating body of nodes.

        Returns:
            np.ndarray: View factor.
        """
        r_values = np.array([C.r_earth if body == 'earth' else C.r_sun if body == 'sun' else np.nan for body in r_values])

        # Calculate additional variables needed for view factor calculation
        gamma_rads = np.deg2rad(gammas)
        r_sc = r_values + self.h
        H = r_sc / r_values
        phi_m = np.arcsin(1 / H)
        b = np.sqrt(H ** 2 - 1)

        # Initialize view factor array
        view_factors = np.zeros_like(gamma_rads)

        # Compute view factors for each node
        for i, gamma_rad in enumerate(gamma_rads):
            if np.isnan(gamma_rad) or np.isnan(H[i]) or np.isnan(b[i]):
                view_factors[i] = 0  # Set view factor to 0 for invalid inputs
                continue

            # Check the range of gamma and calculate the view factor accordingly
            if gamma_rad <= (np.pi / 2 - phi_m[i]):
                view_factors[i] = np.cos(gamma_rad) / H[i] ** 2
            elif (np.pi / 2 - phi_m[i]) < gamma_rad <= (np.pi / 2 + phi_m[i]):
                t1 = 1/2 * np.arcsin(b[i] / (H[i] * np.sin(gamma_rad)))
                t2 = 1 / (2 * H[i] ** 2) * (np.cos(gamma_rad) * np.arccos(-b[i] / np.tan(gamma_rad)) - b[i] * np.sqrt(1 - H[i] ** 2 * np.cos(gamma_rad) ** 2))
                view_factors[i] = 2 / np.pi * (np.pi / 4 - t1 + t2)
            else:
                view_factors[i] = 0  # Set view factor to 0 for angles outside the specified range

        return view_factors

    def albedo(self):
        """
        Get the albedo of the earth.

        Returns:
            float: Albedo of the earth.
        """
        return 0.14 if self.beta < 30 else 0.19

    def earth_ir(self):
        """
        Get the earth IR.

        Returns:
            float: Earth IR.
        """
        return 228 if self.beta < 30 else 218


class ExternalHeatFlux:
    """
    Represents the external heat flux on a node in a thermal model.

    Attributes:
        n (Node): The node on which the external heat flux is applied.
        op (OrbitProperties): The orbit properties of the node.
        _eclipse_cache (bool): Cache for eclipse status.
    
    Methods:
        __init__(self, Node, h, beta, t): Initializes an instance of the ExternalHeatFlux class.
        vf_node(self): Calculates the view factor of the node.
        _calculate_eclipse(self): Calculates and caches the eclipse status.
        solar_flux(self): Calculates the solar flux on the node.
        albedo_flux(self): Calculates the earth albedo flux on the node.
        earth_flux(self): Calculates the earth IR flux on the node.
        total_flux(self): Calculates the total external heat flux on the node.
    """

    __slots__ = ['nodes', 'op', '_t', '_eclipse_cache', '_albedo', '_earth_ir', 'areas', 'absorptivities', 'radiating_bodies', 'gammas']

    def __init__(self, nodes, h, beta, t) -> None:
        """
        Initializes an instance of the ExternalHeatFlux class.

        Parameters:
            nodes (dict): The node on which the external heat flux is applied.
            h (float): The altitude of the node.
            beta (float): The angle between the node's normal vector and the sun vector.
            t (float): The time at which the external heat flux is calculated.
        """
        self.nodes = nodes
        self.op = OrbitProperties(h, beta)
        self._t = t
        self._eclipse_cache = self.calculate_eclipse_status()  # Cache for eclipse status

        self._albedo = self.op.albedo()
        self._earth_ir = self.op.earth_ir()
        # Precompute node properties
        self.areas = np.array([node.area for node in self.nodes.values()])
        self.absorptivities = np.array([node.absorptivity for node in self.nodes.values()])
        self.radiating_bodies = np.array([node.radiating_body for node in self.nodes.values()])
        self.gammas = np.array([node.gamma for node in self.nodes.values()])

    @property
    def t(self):
        """
        Gets the current time value.

        Returns:
            float: The current time value.
        """
        return self._t

    @t.setter
    def t(self, value):
        """
        Sets the time value and updates the eclipse status.

        Parameters:
            value (float): The new time value.
        """
        self._t = value
        self._eclipse_cache = self.calculate_eclipse_status()

    def update_altitude(self, new_altitude):
        """
        Update the altitude in the OrbitProperties instance.

        Parameters:
            new_altitude (float): The new altitude value.
        """
        self.op.h = new_altitude  # Update altitude in OrbitProperties
        self._eclipse_cache = None  # Invalidate eclipse cache
    
    def update_beta_angle(self, new_beta_angle):
        """
        Update the beta angle in the OrbitProperties instance.

        Parameters:
            new_beta_angle (float): The new beta angle value.
        """
        self.op.beta = new_beta_angle  # Update beta angle in OrbitProperties
        self._eclipse_cache = None  # Invalidate eclipse cache

    def calculate_eclipse_status(self):
        """
        Calculates the eclipse status for all nodes.

        Returns:
            numpy.ndarray: Array of eclipse status (True/False) for each node.
        """
        # Calculate eclipse status based on the time 't' for all nodes
        return np.array([self.op.eclipse(self.t) for node in self.nodes.values()])

    def solar_flux(self):
        """
        Calculates the solar flux on the node.

        Returns:
            The solar flux on the node.
        """

        if self._eclipse_cache is None:
            self._eclipse_cache = self.calculate_eclipse_status()

        vf = self.op.view_factor(self.gammas, self.radiating_bodies)
        return np.where(self._eclipse_cache is not None and ~self._eclipse_cache, vf * self.areas * C.q_sun * self.absorptivities, 0)

    def albedo_flux(self):
        """
        Calculates the earth albedo flux on the node.

        Returns:
            The earth albedo flux on the node.
        """
        if self._eclipse_cache is None:
            self._eclipse_cache = self.calculate_eclipse_status()

        return np.where(self._eclipse_cache is not None and ~self._eclipse_cache, self._albedo * self.areas * C.q_sun * self.absorptivities, 0)

    def earth_flux(self):
        """
        Calculates the earth IR flux on the node.

        Returns:
            The earth IR flux on the node.
        """
        return self._earth_ir * self.areas

    def total_flux(self):
        """
        Calculates the total external heat flux on each node.

        Returns:
            np.ndarray: Array of total external heat flux for each node.
        """
        solar_flux = self.solar_flux()
        albedo_flux = self.albedo_flux()
        earth_flux = self.earth_flux()
        return solar_flux + albedo_flux + earth_flux
   

class ThermalModel:
    """
    A class representing a thermal model.

    Attributes:
        nodes (list): List of nodes in the thermal model.

    Methods:
        __init__(self, nodes: list) -> None: Initializes the ThermalModel object.
        add_node(self, node: Node): Adds a node to the list of nodes.
        remove_node(self, key): Removes a node from the list of nodes.
        get_node(self, key): Retrieves a node by its key.
        normal_vector_from_angles(theta_xy, theta_yz, theta_xz): Calculates the normal vector from given angles.
        calculate_angles(node_i, node_j, normal1, normal2): Calculates the angles between the normal vectors of two faces.
        compute_view_factor(args): Computes the view factor between two nodes.
        is_occluded(self, index_i, index_j, positions): Checks if any node occludes the view between two nodes.
        internal_vf(self): Calculates the view factor between different nodes.
        compute_conductivity_matrix(self): Calculates the conductivity matrix.
        compute_area_matrix(self): Calculates the area matrix.
        compute_distance_matrix(self): Calculates the distance matrix.
        heat_balance(self, h, beta, t): Calculates the heat balance equation.
        integrate_heat_balance(self, beta_range, h_range, time_range): Integrates the heat balance equation over a range of parameters.
    """

    __slots__ = ['nodes', 'vf_matrix', 'k_matrix', 'a_matrix', 'l_matrix']

    def __init__(self, nodes: list) -> None:
        """
        Initializes a ThermalModel object.

        Parameters:
            nodes (list): List of nodes in the thermal model.
            vf_matrix (numpy.ndarray): View factor matrix.
            k_matrix (numpy.ndarray): Conductivity matrix.
            a_matrix (numpy.ndarray): Area matrix.
            l_matrix (numpy.ndarray): Distance matrix.
        """
        self.nodes = {node.key: node for node in nodes}
        self.vf_matrix = self.internal_vf()
        self.k_matrix = self.compute_conductivity_matrix()
        self.a_matrix = self.compute_area_matrix()
        self.l_matrix = self.compute_distance_matrix()

    def add_node(self, node: Node) -> None:
        """
        Adds a node to the list of nodes in the thermal model.

        Parameters:
            node (Node): The node to be added.
        """
        self.nodes[node.key] = node
        self.vf_matrix = self.internal_vf()
        self.k_matrix = self.compute_conductivity_matrix()
        self.a_matrix = self.compute_area_matrix()
        self.l_matrix = self.compute_distance_matrix()

    def remove_node(self, key: str) -> None:
        """
        Removes a node from the list of nodes based on the given key.

        Parameters:
            key (str): The key of the node to be removed.
        """
        if key in self.nodes:
            del self.nodes[key]
            self.vf_matrix = self.internal_vf()
            self.k_matrix = self.compute_conductivity_matrix()
            self.a_matrix = self.compute_area_matrix()
            self.l_matrix = self.compute_distance_matrix()

    def get_node(self, key: str) -> Node:
        """
        Get a node from the thermal model by its key.

        Parameters:
            key (str): The key of the node to retrieve.

        Returns:
            Node or None: The node with the specified key, or None if not found.
        """
        return self.nodes.get(key, None)
    
    @staticmethod
    def normal_vector_from_angles(theta_xy: float, theta_yz: float, theta_xz: float) -> np.ndarray:
        """
        Calculate the normal vector from given angles.

        Parameters:
            theta_xy (float): Angle in degrees between the x-y plane and the normal vector.
            theta_yz (float): Angle in degrees between the y-z plane and the normal vector.
            theta_xz (float): Angle in degrees between the x-z plane and the normal vector.

        Returns:
            numpy.ndarray: Normal vector with components in x, y, and z directions.
        """
        # Calculate direction cosines
        cos_theta_xy = np.cos(np.radians(theta_xy))
        cos_theta_yz = np.cos(np.radians(theta_yz))
        cos_theta_xz = np.cos(np.radians(theta_xz))

        normal_vector = np.array([cos_theta_xy, cos_theta_yz, cos_theta_xz])
        normal_vector /= np.linalg.norm(normal_vector)
        return normal_vector

    @staticmethod
    def calculate_angles(node_i: Node, node_j: Node, normal1: np.ndarray, normal2: np.ndarray) -> tuple:
        """
        Calculate the angles between the normal vectors of two faces.

        Parameters:
            node_i (Node): The first node.
            node_j (Node): The second node.
            normal1 (numpy.ndarray): Normal vector of the first node.
            normal2 (numpy.ndarray): Normal vector of the second node.

        Returns:
            tuple: Tuple containing the following values:
                theta_i (float): Angle between the normal vector of the first node and the vector connecting the two nodes.
                theta_j (float): Angle between the normal vector of the second node and the vector connecting the two nodes.
                R_ij (numpy.ndarray): Vector connecting the centers of the two faces.
        """
        # Calculate the vector connecting the centers of the two faces
        R_ij = np.array(node_i.position[0]) - np.array(node_j.position[0])

        # Normalize the vector connecting the centers
        vector_ij_normalized = R_ij / np.linalg.norm(R_ij)

        # Calculate the dot product of the connecting vector with the normal vectors
        dot_product1 = np.dot(vector_ij_normalized, normal1)
        dot_product2 = np.dot(-vector_ij_normalized, normal2)  # Reverse direction for face2

        # Calculate the angles using the arccosine of the dot product
        theta_i = np.degrees(np.arccos(dot_product1))
        theta_j = np.degrees(np.arccos(dot_product2))

        return theta_i, theta_j, R_ij

    @staticmethod
    def compute_view_factor(args) -> float:
        """
        Compute the view factor between two nodes.

        Parameters:
            args (tuple): Tuple containing the following arguments:
                node_i (Node): The first node.
                node_j (Node): The second node.
                normal1 (numpy.ndarray): Normal vector of the first node.
                normal2 (numpy.ndarray): Normal vector of the second node.
                A_i (float): Area of the first node.

        Returns:
            float: The view factor between the two nodes.
        """
        node_i, node_j, normal1, normal2, A_i = args
        theta_i, theta_j, R_ij = ThermalModel.calculate_angles(node_i, node_j, normal1, normal2)
        R_ij_norm = np.linalg.norm(R_ij)

        def integrand(x, y, A_i, A_j, R_ij_norm, theta_i, theta_j):
            return np.cos(np.deg2rad(theta_i)) * np.cos(np.deg2rad(theta_j)) / (np.pi * R_ij_norm**2)

        x_i_min, x_i_max = -np.sqrt(A_i)/2, np.sqrt(A_i)/2
        y_j_min, y_j_max = -np.sqrt(node_j.area)/2, np.sqrt(node_j.area)/2

        result, _ = dblquad(
            integrand,
            x_i_min, x_i_max,
            lambda x: y_j_min, lambda x: y_j_max,
            args=(A_i, node_j.area, R_ij_norm, theta_i, theta_j)
        )

        return result / A_i

    def is_occluded(self, index_i: int, index_j: int, positions: np.ndarray) -> bool:
        """
        Check if any node occludes the view between two nodes.

        Parameters:
            index_i (int): Index of the first node.
            index_j (int): Index of the second node.
            positions (numpy.ndarray): Array of node positions.

        Returns:
            bool: True if the view is occluded, False otherwise.
        """
        position_i = positions[index_i]
        position_j = positions[index_j]

        for k, position_k in enumerate(positions):
            if k != index_i and k != index_j:
                # Simple occlusion check along straight lines (x, y, z axes)
                if all(position_i <= position_k) and all(position_k <= position_j):
                    return True  # Node k occludes the view between node i and j

        return False

    def internal_vf(self) -> np.ndarray:
        """
        Calculate the view factor matrix between different nodes.

        Returns:
            numpy.ndarray: View factor matrix.
        """
        node_count = len(self.nodes)
        vf_matrix = np.zeros((node_count, node_count))  # Initialize matrix
        nodes_list = list(self.nodes.values())

        # Precompute normal vectors and positions
        normal_vectors = [self.normal_vector_from_angles(*node.position[1]) for node in nodes_list]
        positions = np.array([node.position[0] for node in nodes_list])
        areas = np.array([node.area for node in nodes_list])

        for i, node_i in enumerate(nodes_list):
            normal_i = normal_vectors[i]
            position_i = positions[i]
            area_i = areas[i]

            for j, node_j in enumerate(nodes_list):
                if i != j:
                    normal_j = normal_vectors[j]
                    position_j = positions[j]

                    # Check if any node is occluding the view between node_i and node_j
                    if self.is_occluded(i, j, positions):
                        vf_matrix[i][j] = 0
                        continue

                    R_ij = position_i - position_j
                    R_ij_norm = np.linalg.norm(R_ij)

                    # Calculate the dot product of the connecting vector with the normal vectors
                    vector_ij_normalized = R_ij / R_ij_norm
                    dot_product1 = np.dot(vector_ij_normalized, normal_i)
                    dot_product2 = np.dot(-vector_ij_normalized, normal_j)

                    # Calculate the angles using the arccosine of the dot product
                    theta_i_rad = np.arccos(dot_product1)
                    theta_j_rad = np.arccos(dot_product2)
                    theta_i = np.degrees(theta_i_rad)
                    theta_j = np.degrees(theta_j_rad)

                    def integrand(x, y, A_i, A_j, R_ij_norm, theta_i, theta_j):
                        return np.cos(np.deg2rad(theta_i)) * np.cos(np.deg2rad(theta_j)) / (np.pi * R_ij_norm**2)

                    x_i_min, x_i_max = -np.sqrt(area_i)/2, np.sqrt(area_i)/2
                    y_j_min, y_j_max = -np.sqrt(node_j.area)/2, np.sqrt(node_j.area)/2

                    result, _ = dblquad(
                        integrand,
                        x_i_min, x_i_max,
                        lambda x: y_j_min, lambda x: y_j_max,
                        args=(area_i, node_j.area, R_ij_norm, theta_i, theta_j)
                    )

                    vf_matrix[i][j] = result / area_i

        return vf_matrix
    
    def compute_conductivity_matrix(self) -> np.ndarray:
        """
        Calculate the conductivity matrix.

        Returns:
            numpy.ndarray: Conductivity matrix.
        """
        node_count = len(self.nodes)

        node_indices = {node_key: idx for idx, node_key in enumerate(self.nodes.keys())}

        k_matrix = np.zeros((node_count, node_count))
        for i, (key_i, node_i) in enumerate(self.nodes.items()):
            for neighbor_node, _ in node_i.get_neighbors():
                j = node_indices[neighbor_node.key]
                k_matrix[i, j] = neighbor_node.conductivity
                k_matrix[i, i] -= neighbor_node.conductivity

        return k_matrix
    
    def compute_area_matrix(self) -> np.ndarray:
        """
        Calculate the area matrix.

        Returns:
            numpy.ndarray: Area matrix.
        """
        node_count = len(self.nodes)

        node_indices = {node_key: idx for idx, node_key in enumerate(self.nodes.keys())}

        area_matrix = np.zeros((node_count, node_count))
        for i, (key_i, node_i) in enumerate(self.nodes.items()):
            for neighbor_node, contact_area in node_i.get_neighbors():
                j = node_indices[neighbor_node.key]
                area_matrix[i, j] = contact_area
        
        return area_matrix
    
    def compute_distance_matrix(self) -> np.ndarray:
        """
        Calculate the distance matrix.

        Returns:
            numpy.ndarray: Distance matrix.
        """
        node_count = len(self.nodes)

        node_indices = {node_key: idx for idx, node_key in enumerate(self.nodes.keys())}

        distance_matrix = np.zeros((node_count, node_count))
        for i, (key_i, node_i) in enumerate(self.nodes.items()):
            for neighbor_node, _ in node_i.get_neighbors():
                j = node_indices[neighbor_node.key]
                distance_matrix[i, j] = np.linalg.norm(node_i.position[0] - neighbor_node.position[0])
        
        return distance_matrix

    def heat_balance(self, h: float, beta: float, t: float) -> np.ndarray:
        """
        Calculate the rate of change of temperature for each node in the thermal model.

        Parameters:
            h (float): Convective heat transfer coefficient.
            beta (float): Solar absorptivity.
            t (float): Time.

        Returns:
            dT_dt (numpy.ndarray): Array of the rate of change of temperature for each node.
        """
        thermal_control = False

        node_indices = {node_key: idx for idx, node_key in enumerate(self.nodes.keys())}

        temperatures = np.array([node.temperature for node in self.nodes.values()])
        # Call each node's get_emissivity method to account for MLI
        emissivities = np.array([node.get_emissivity() if hasattr(node, 'mli') and node.mli else node.emissivity for node in self.nodes.values()])
        areas = np.array([node.area for node in self.nodes.values()])
        thermal_masses = np.array([node.thermal_mass for node in self.nodes.values()])
        heat_flux_ints = np.array([node.heat_flux_int for node in self.nodes.values()])

        temperature_differences = np.zeros_like(self.a_matrix)
        for i, node_i in enumerate(self.nodes.values()):
            for neighbor_key in node_i.neighbors:
                j = node_indices[neighbor_key]
                if i != j:
                    temperature_differences[i, j] = temperatures[j] - temperatures[i]

        with np.errstate(divide='ignore', invalid='ignore'):
            heat_transfer_matrix = np.where(self.l_matrix != 0, self.k_matrix * temperature_differences * self.a_matrix / self.l_matrix, 0)

        q_internal_conducted = np.sum(heat_transfer_matrix, axis=1)

        temp_diff = temperatures[:, np.newaxis] ** 4 - temperatures[np.newaxis, :] ** 4
        # Update q_internal_radiated to use the new emissivities from MLI where applicable
        q_internal_radiated = np.sum(emissivities[:, np.newaxis] * C.sigma * temp_diff * areas[:, np.newaxis] * self.vf_matrix, axis=1)

        radiating_bodies = np.array([node.radiating_body for node in self.nodes.values()])
        is_radiating = np.isin(radiating_bodies, ['earth', 'sun'])
        # Update q_radiated to use the new emissivities from MLI where applicable
        q_radiated = np.where(is_radiating, emissivities * C.sigma * areas * (temperatures ** 4 - C.T_space ** 4), 0)

        q_generated = np.array([node.heat_flux_int for node in self.nodes.values()])
        
        external_heat_flux = ExternalHeatFlux(self.nodes, h, beta, t).total_flux()

        q_external = np.where(is_radiating, external_heat_flux, 0)

        q_total = q_internal_conducted + q_generated + q_internal_radiated - q_radiated + q_external

        #print(np.max(q_internal_conducted), np.max(q_generated), np.max(q_internal_radiated), np.max(q_radiated), np.max(q_external))
        #print(np.min(q_internal_conducted), np.min(q_generated), np.min(q_internal_radiated), np.min(q_radiated), np.min(q_external))

        #print(f'{self.nodes[1].name} : {q_total[0]}, {self.nodes[2].name} : {q_total[1]}, {self.nodes[3].name} : {q_total[2]}, {self.nodes[4].name} : {q_total[3]}, {self.nodes[9].name} : {q_total[8]}')

        # print(np.max(q_total), np.min(q_total))
        if thermal_control:
            dT_dt = np.where(heat_flux_ints == 0, q_total / thermal_masses, 0)
            # print(f'Thermal control {dT_dt}')
        else:
            dT_dt = q_total / thermal_masses
            # print(dT_dt)
        return dT_dt

    @staticmethod
    def integrate_one_scenario(args) -> np.ndarray:
        """
        Integrate the heat balance equation for a single scenario.

        Parameters:
            args (tuple): Tuple containing the following arguments:
                beta (float): Solar absorptivity.
                h (float): Convective heat transfer coefficient.
                time (float): Time.
                initial_T (numpy.ndarray): Array of initial temperature values for each node.
                ode_system_wrapper (function): Function that calculates the rate of change of temperature for each node in the thermal model.

        Returns:
            numpy.ndarray: Array of temperature values for each node.
        """
        beta, h, time, initial_T, ode_system_wrapper = args
        ode_system = ode_system_wrapper(h, beta)
        sol = solve_ivp(ode_system, [0, time], initial_T, method='RK45')
        return sol.y[:, -1]

    def ode_system_wrapper(self, h: float, beta: float):
        """
        Returns a function that calculates the rate of change of temperature for each node in the thermal model.

        Parameters:
            h (float): Convective heat transfer coefficient.
            beta (float): Solar absorptivity.

        Returns:
            function: Function that calculates the rate of change of temperature for each node in the thermal model.
        """
        def ode_system(t: float, y: np.ndarray) -> np.ndarray:
            """
            ode_system calculates the rate of change of temperature for each node in the thermal model.

            Parameters:
                t (float): Time.
                y (numpy.ndarray): Array of temperature values for each node.

            Returns:
                numpy.ndarray: Array of the rate of change of temperature for each node.
            """
            for node, temp in zip(self.nodes.values(), y):
                node.update_temperature(temp)
            return self.heat_balance(h, beta, t)

        return ode_system

    def integrate_heat_balance(self, beta_range: np.ndarray, h_range: np.ndarray, time_range: np.ndarray) -> np.ndarray:
        """
        integrate the heat balance equation over a range of parameters.

        Parameters:
            beta_range (numpy.ndarray): Array of beta angles (in degrees).
            h_range (numpy.ndarray): Array of altitudes.
            time_range (numpy.ndarray): Array of time values.

        Returns:
            results (numpy.ndarray): Array of temperature values for each node.
        """
        initial_T = [node.temperature for node in self.nodes.values()]
        results_shape = (len(beta_range), len(h_range), len(time_range), len(initial_T))
        results = np.zeros(results_shape)

        # Prepare arguments for parallel processing
        all_args = [(beta, h, time, initial_T, self.ode_system_wrapper) 
                    for beta in beta_range for h in h_range for time in time_range]

        with Pool() as pool:
            with tqdm(total=len(all_args), desc="Integrating Heat Balance") as pbar:
                for index, result in enumerate(pool.imap_unordered(ThermalModel.integrate_one_scenario, all_args)):
                    i = index // (len(h_range) * len(time_range))
                    k = (index // len(time_range)) % len(h_range)
                    j = index % len(time_range)
                    results[i, k, j, :] = result
                    pbar.update(1)

        return results
