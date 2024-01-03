"""
A module that contains various constants used in the thermal calculations.
"""

class Constants:
    """
    A class that contains various constants used in the thermal calculations.
    """
    G = 6.674 * (10 ** -11)  # Gravitational constant in m^3 kg^-1 s^-2
    M = 5.972 * (10 ** 24)   # Mass of Earth in kg
    mu = G * M  # Earth's gravitational parameter in m^3 s^-2
    r_earth = 6378 # radius earth km
    r_sun = 147310000 # Distance earth-sun km

    A_panel = 4.657 # Area side skin panel m^2
    A_top = 2.25 # Area top skin panel m^2
    A_solar = 6 # Area solar array m^2
    A_compartment = 2.64 # Radiating area pressurized compartment m^2

    A_contact_boom = 0.00149 # Contact area boom-skin m^2 thin walled 5 cm diameter
    A_contact_side = 0.025 # Contact area side skin panels m^2 0.005 (t) * 5 (height)
    A_contact_top = 0.0279 # Contact area top skin panel m^2 0.005 (t) * 0.93 (width panel) * 6 (sides)
    A_contact_support = 0.001 # Contact area compartment support m^2

    q_sun = 1370 # Solar intensity sun W / m^2

    m_panel = 70 # Mass skin side panel kg 0.005 (t) * 0.93 (w) * 5 (h) * 2710 (rho)
    m_top = 30 # Mass top skin panel kg
    m_array = 60 # Mass solar arrays kg each

    cp_al = 897 # Specific heat capacity aluminium J / kg K
    cp_ga = 340 # Specific heat capacity gallium J / kg K

    k_al = 237 # Thermal conductivity aluminium W / m K
    k_ga = 40 # Thermal conductivity gallium W / m K

    # absorptance and emmitance of white paint
    alpha_external = 0.3 # absorptance of outside of skin panels
    alpha_internal = 0.4 # absorptance of inside of skin panels
    alpha_solar = 0.8 # absorptance of solar panels
    epsilon = 0.9 #external skin panels
    epsilon_solar = 0.8 # emmitance solar panels
    epsilon_compartment = 0.31 # emmitance aluminium

    sigma = 5.67 * 10 ** -8 # Stefan-Boltzmann constant W / m^2 K^4

    T_0 = 293.15 # Initial temperature K
    T_space = 2.725 # Temperature of space K
