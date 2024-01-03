"""
A module that contains constructors for materials and components.
"""


class Material:
    """
    Represents a material with its properties.
    
    Parameters:
        name (str): The name of the material.
        conductivity (float): The thermal conductivity of the material.
        specific_heat (float): The specific heat capacity of the material.
        density (float): The density of the material.
    """
    def __init__(self, name, conductivity, specific_heat, emissivity, absorptivity):
        self.name = name
        self.conductivity = conductivity
        self.cp = specific_heat
        self.emissivity = emissivity
        self.absorptivity = absorptivity


class Component(Material):
    """
    Represents a component with its properties.
    
    Parameters:
        name (str): The name of the component.
        power (float): The power of the component.
        material (Material): The material of the component.
        dimensions (tuple): The dimensions of the component.
        efficiency (float): The efficiency of the component.
    """
    def __init__(self, name, power, material, efficiency):
        super().__init__(material.name, material.conductivity, material.cp, material.emissivity, material.absorptivity)
        self.name = name
        self.power = power
        self.efficiency = efficiency
