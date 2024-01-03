from multiprocessing import Pool
import numpy as np
from constants import Constants as C


class Body:
    """
    Body class for storing information about a celestial body.
    """
    
    __slots__ = ['key', 'name', 'mass', 'radius', 'position', 'velocity', 'acceleration', 'force', 'positions']
    
    def __init__(self, key: int, name: str, mass: float, radius: float, position: list, velocity: list, acceleration: list, force: list) -> None:
        """
        Body class for storing information about a celestial body.
        
        Parameters:
            key (int): Unique identifier for the body.
            name (str): Name of the body.
            mass (float): Mass of the body in kg.
            radius (float): Radius of the body in m.
            position (list): Position of the body in m.
            velocity (list): Velocity of the body in m/s.
            acceleration (list): Acceleration of the body in m/s^2.
            force (list): Force acting on the body in N.
            neighbors (dict): Dictionary of neighboring bodies.
        """
        self.key = key
        self.name = name
        self.mass = mass
        self.radius = radius
        self.position = np.array(position, dtype=np.float64)
        self.velocity = np.array(velocity, dtype=np.float64)
        self.acceleration = np.array(acceleration, dtype=np.float64)
        self.force = force
        self.positions = {0: self.position.copy()}
        
    def __str__(self) -> str:
        """
        A string representation of the body.
        
        Returns:
            str: A string representation of the body.
        """
        return f'{self.name} ({self.key})'
        
    def __repr__(self) -> str:
        """
        A string representation of the body.
        
        Returns:
            str: A string representation of the body.
        """
        return f'{self.name} ({self.key})'

    def update_position(self, time_step: float, current_time: float) -> None:
        """
        Updates the position of the body based on its velocity.

        Parameters:
            time_step (float): Time step in seconds.
            current_time (float): Current time in seconds.
        """
        self.velocity = self.velocity.astype(np.float64)
        self.position += self.velocity * float(time_step)
        # print(f'Position: {self.position}, Velocity: {self.velocity} for {self.name} at {current_time} seconds')
        self.positions[int(current_time + time_step)] = self.position.copy()
        # print(f'Position history: {self.positions[current_time + time_step]} for {self.name} at {current_time + time_step} seconds')

    @staticmethod
    def gravitational_force(body1, body2) -> np.array:
        """
        Calculates the gravitational force between two bodies.

        Parameters:
            body1 (Body): First body.
            body2 (Body): Second body.
        
        Returns:
            Vector: Gravitational force vector between the two bodies.
        """
        distance_vector = body2.position - body1.position
        distance = np.linalg.norm(distance_vector)
        if distance < 1e-10:  # Prevent division by zero with a small threshold
            return np.zeros_like(distance_vector)

        force_magnitude = C.G * body1.mass * body2.mass / distance**2
        force_vector = force_magnitude * distance_vector / distance
        return force_vector

    def apply_gravity(self, other_bodies: list, time_step: float) -> None:
        """
        Applies the gravitational force of other bodies to this body.

        Parameters:
            other_bodies (list): List of other bodies in the space.
            time_step (float): Time step in seconds.
        """
        total_force = np.sum([self.gravitational_force(self, other) for other in other_bodies if other != self], axis=0)
        # Update velocity based on the net force
        self.acceleration = total_force / self.mass
        # print(f'{self.acceleration} for {self.name}')
        self.velocity = self.velocity.astype(np.float64)
        self.velocity += self.acceleration * time_step
        # print(f'Velocity: {self.velocity} for {self.name}')
        
    def get_position_at_time(self, time: float) -> np.array:
        """
        Returns the position of the body at a given time.

        Parameters:
            time (float): Time in seconds.
        
        Returns:
            Vector: Position of the body at the given time.
        """
        return self.positions[time]
    
   
class Space:
    """
    class for storing information about the space in which the bodies exist.
    """
    __slots__ = ['bodies']
    
    def __init__(self, bodies: dict) -> None:
        """
        class for storing information about the space in which the bodies exist.
        
        Parameters:
            bodies (dict): Dictionary of bodies in the space.
            time (float): Time of the space in seconds.
        """
        self.bodies = bodies
        
    def __str__(self) -> str:
        """
        A string representation of the space.
        
        Returns:
            str: A string representation of the space.
        """
        return f'Space with {len(self.bodies)} bodies'
    
    def __repr__(self) -> str:
        """
        A string representation of the space.
        
        Returns:
            str: A string representation of the space.
        """
        return f'Space with {len(self.bodies)} bodies'
    
    def __getitem__(self, key: int) -> Body:
        """
        Get a body from the space.
        
        Parameters:
            key (int): The key of the body to get.
            
        Returns:
            Body: The body with the given key.
        """
        return self.bodies[key]
    
    def __setitem__(self, key: int, value: Body) -> None:
        """
        Set a body in the space.
        
        Parameters:
            key (int): The key of the body to set.
            value (Body): The body to set.
        """
        self.bodies[key] = value
        
    def __delitem__(self, key: int) -> None:
        """
        Delete a body from the space.
        
        Parameters:
            key (int): The key of the body to delete.
        """
        del self.bodies[key]

    def update(self, time_step: float, current_time: float) -> None:
        """
        Updates the positions of the bodies in the space.

        Parameters:
            time_step (float): Time step in seconds.
            current_time (float): Current time in seconds.
        """
        for body in self.bodies:
            body.apply_gravity(self.bodies, time_step)
            body.update_position(time_step, current_time)

    def simulate(self, total_time: float, time_step: float, parallel=False) -> None:
        """
        Simulates the space for a given amount of time.

        Parameters:
            total_time (float): Total time to simulate in seconds.
            time_step (float): Time step in seconds.
            parallel (bool): Whether to use multiprocessing.
        """
        current_time = 0
        steps = int(total_time / time_step)
        if parallel:
            for _ in range(steps):
                self.update_parallel(time_step, current_time)
                current_time += time_step
        for _ in range(steps):
            self.update(time_step, current_time)
            current_time += time_step

    def apply_gravity_parallel(self, body_index: int, time_step: float) -> Body:
        """
        Applies the gravitational force of other bodies to this body.

        Parameters:
            body_index (int): Index of the body to apply gravity to.
            time_step (float): Time step in seconds.
        """
        body = self.bodies[body_index]
        body.apply_gravity(self.bodies, time_step)
        return body

    def update_parallel(self, time_step: float, current_time: float) -> None:
        """
        Updates the positions of the bodies in the space using multiprocessing.

        Parameters:
            time_step (float): Time step in seconds.
            current_time (float): Current time in seconds.
        """
        with Pool() as pool:
            updated_bodies = pool.starmap(self.apply_gravity_parallel, [(i, time_step) for i in range(len(self.bodies))])
            for _, body in enumerate(updated_bodies):
                body.update_position(time_step, current_time)
        self.bodies = updated_bodies
