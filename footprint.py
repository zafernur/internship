import numpy as np
from scipy import special, optimize
import matplotlib.pyplot as plt
from scipy.integrate import simps

"""
This module contains some functions required for footprint calculations, which are used to make necessary
corrections in cases where the gamma-ray source is not a point source but is distributed throughout a volume.
The functions in this module were meant to use for a salt mine project. Therefore; some fucntions and constants
may needed to be changed for a given geometry.
"""

# Constants

A = 1                   # Area of the crystal that is directed to the source volume [m^2].
Eps = 1                 # The efficiency of the crystal.
Gamma = 1               # Number of gamma-rays needed to create a count.
max_source_width = 1.25 # For the salt mine project, this is the height of the wall that contains the source.
                        # It depends on the geometry [m].


def TotalGamma(distance, rho_a, mu_a, rho_w, mu_w):
    """
    Calculates the total contribution of a volume of source at a distance.

    Args:
        distance (float): The distance from the crystal to the wall (volume of source) [m]
        rho_a (float): Density of air [kg/m^3]
        rho_w (float): Density of the wall [kg/m^3]
        mu_a (float): Mass attenuation coefficient of air [m^2/kg]
        mu_w (float): Mass attenuation coefficient of the wall [m^2/kg]

    Returns:
        float: Total gamma-ray contribution that is radiated by the wall.
    """
    x = mu_a*rho_a*distance
    prefix = ( (A*Eps*Gamma)/(2*mu_w*rho_w) )
    out = prefix * special.expn(2,x)

    return out

def depth_into_wall(distance, percentage, rho_a, mu_a, rho_w, mu_w):
    """
    Calculates the maximum depth in to the wall across the crsytal.

    Args:
        distance (float): The distance from the crystal to the wall (volume of source) [m]
        percentage (float): The percentage of signal that stems from a certain amount of volume.
        rho_a (float): Density of air [kg/m^3]
        rho_w (float): Density of the wall [kg/m^3]
        mu_a (float): Mass attenuation coefficient of air [m^2/kg]
        mu_w (float): Mass attenuation coefficient of the wall [m^2/kg]

    Returns:
        float: The maximum depth into the wall at the level of the signal [m].
    """
    lower_lim = mu_a * rho_a* distance
    integral = special.expn(2,lower_lim)
    outer_volume_contribution = (1-percentage/100) * integral
    depth = (-1/rho_w) * ((np.log(outer_volume_contribution)/mu_w) + distance * rho_a)
    
    return depth

def getIsolines(distance, percentages, rho_a, mu_a, rho_w, mu_w, theta_max):
    """
    Calculates the points of the isolines that define the boundaries of the source of volume
    from which the given signal percentage originates.

    Args:
        distance (float): The distance from the crystal to the wall (volume of source) [m]
        percentage (float): The percentage of signal that stems from a certain amount of volume.
        rho_a (float): Density of air [kg/m^3]
        rho_w (float): Density of the wall [kg/m^3]
        mu_a (float): Mass attenuation coefficient of air [m^2/kg]
        mu_w (float): Mass attenuation coefficient of the wall [m^2/kg]
        theta_max (float): The angle between the line that points towards the wall
                           from the crystal and the line points towards to the furthest
                           end point of the source of volume. The arctan value of the
                           ratio of the maximum wall height to the distance of the crystal
                           from the wall [degree].

    Returns:
        Tuple: The elements of the tuple: 1. The x values of the isoline curves.
                                          2. Broadcasted zeros for using to plot the wall.
                                          3. The height points that construct the wall.
    """
    limits = np.deg2rad(theta_max)                 
    theta = np.linspace(-1*limits, limits, 3600)

    # Depths into the wall for each percentage
    depths = [depth_into_wall(distance, percentage, rho_a, mu_a, rho_w, mu_w) for percentage in percentages]

    # Isoline points for each percentage
    isoline_points = np.zeros((len(depths), len(theta)))
    for i in range(len(depths)):
        isoline_points[i] = ((rho_a/rho_w) * distance * (1 - (1 / np.cos(theta))) + depths[i]) * -np.cos(theta)
        for j in range(len(isoline_points[i])):
            if isoline_points[i][j] > 0:
                isoline_points[i][j] = 0;
    
    # The boundary (The wall or the ground)
    boundary_x = 0 * theta
    boundary_y = distance * np.tan(theta)

    return isoline_points, boundary_x, boundary_y

def signal_volume(isolines, points_on_wall, percentages, seen_radius, boundary, distance, crystal_diameter):
    """
    Calculates the volume, from which the given percentage of the signal originates,
    bounded by the isoline and the outermost radii seen by the crsytal.

    Args:
        isolines (array): The x points of the isoline for a corresponded percentage of signal.
        points_on_wall (array): The height points that construct the wall.
        percentages (list): The percentages of signal that stems from a certain amount of volume.
        seen_radius (float): The maximum height value on the wall that is seen by the crystal [m].
        boundary (str): Indicates if the seen_radius is inner or outer limit. This is for a case
                        that a shielding is used. It can take the values: 'inner' or 'outer' 
        distance (float): The distance from the crystal to the wall (volume of source) [m]
        percentage (float): The percentage of signal that stems from a certain amount of volume.
        rho_a (float): Density of air [kg/m^3]
        rho_w (float): Density of the wall [kg/m^3]
        mu_a (float): Mass attenuation coefficient of air [m^2/kg]
        mu_w (float): Mass attenuation coefficient of the wall [m^2/kg]
        theta_max (float): The angle between the line that points towards the wall
                           from the crystal and the line points towards to the furthest
                           end point of the source of volume. The arctan value of the
                           ratio of the maximum wall height to the distance of the crystal
                           from the wall [degree]

    Returns:
        List: The volume(s) from which the given percentage(s) of the signal originates.
    """
    volumes = []
    if seen_radius < max_source_width:
        for i in range(len(percentages)):
            if boundary == 'inner':
                slope = seen_radius / distance
            elif boundary == 'outer':
                slope = (seen_radius + (crystal_diameter / 2)) / distance
    
            differences = np.abs(points_on_wall[1800:] - (slope * (-isolines[i][1800:]) + seen_radius))
            min_index = np.argmin(differences)
            intersection_x = -isolines[i][1800:][min_index]
            intersection_y = points_on_wall[1800:][min_index]
    
            # The region under the line of given radius (boundary): V1
            y1 = points_on_wall[np.argmin(np.abs(points_on_wall - seen_radius)):np.argmin(np.abs(points_on_wall - intersection_y))]
            x1 = (y1 - seen_radius) / slope
            y1_squared = y1 ** 2
    
            # The region under the isoline: V2
            y2 = points_on_wall[1800:np.argmin(np.abs(points_on_wall - intersection_y))]
            x2 = -isolines[i][1800:np.argmin(np.abs(points_on_wall - intersection_y))]
            y2_squared = y2 ** 2
    
            V1 = np.pi * simps(y1_squared, x1)
            V2 = np.pi * simps(y2_squared, np.flip(x2))
    
            if min(x2) - max(x1) > 0.003:
                V3 = np.pi * (min(x2) - max(x1)) * (1.25 ** 2) 
                volumes.append(V1 + V2 + V3)
            else:
                volumes.append(V1 + V2)
    else:
        for i in range(len(percentages)):
            print('seen radius is higher than wall')
            V1 = np.pi * (-max(isolines[i])) * (1.25 ** 2)
            V2 = np.pi * simps(points_on_wall[1800:] ** 2, np.flip(-isolines[i][1800:]))
            volumes.append(V1 + V2)


    print(f'For a shielded 2x{int(crystal_diameter/0.0254)} crsytal, {distance} m away from the wall,')
    print(f'the volumes between {boundary} boundaries that account for 65%, 95% and 99.7% of the signal')
    print(f'are {volumes[0]}, {volumes[1]}, and {volumes[2]} [m^3], respectively.')
    
    return volumes