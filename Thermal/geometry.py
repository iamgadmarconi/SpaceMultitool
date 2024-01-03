import numpy as np

# Constants for the hexagonal prism
height = 5.0  # Height of the hexagonal prism
internal_radius = 1.8  # Internal radius of the hexagonal prism

# Coordinates of the geometric center of the hexagonal prism
geometric_center = (0, 0, height / 2)

# Calculate the coordinates of the centers of the side faces
# The geometric center of the hexagon is at the origin (0, 0)
side_face_centers = [
    (internal_radius * np.cos(np.radians(angle)), internal_radius * np.sin(np.radians(angle)), height / 2)
    for angle in range(0, 360, 60)
]

# Calculate the normal vector angles for the side faces
# The normal vector of each side face will have the same xy-angle as the center of the face
# since it's perpendicular to the face and the face lies along a radius of the hexagon.
# theta_xy is the angle in the xy-plane from the x-axis
# theta_yz is always 90 degrees because the side faces are vertical
# theta_xz is always 90 degrees because the side faces are vertical
side_face_normals = [
    (angle, 90, 90) for angle in range(0, 360, 60)
]

# Coordinates and angles for the top and bottom faces
# The top and bottom faces are parallel to the xy-plane, so their normals are along the z-axis
top_face_center = (0, 0, height)
bottom_face_center = (0, 0, 0)
top_bottom_face_normals = [
    # theta_xy is 0 for top face because its normal points along the positive z-axis
    # theta_yz is 0 because it's parallel to the yz-plane
    # theta_xz is 0 because it's parallel to the xz-plane
    (0, 0, 0), 
    # theta_xy is 180 for bottom face because its normal points along the negative z-axis
    # theta_yz is 180 because it's parallel to the yz-plane but points in the opposite direction
    # theta_xz is 180 because it's parallel to the xz-plane but points in the opposite direction
    (180, 180, 180)
]

# Combine the information for all faces
face_centers = side_face_centers + [top_face_center, bottom_face_center]
face_normals = side_face_normals + top_bottom_face_normals

face_centers, face_normals


# Calculate the direction cosines (angles) for the side faces
def calculate_direction_cosines(center):
    """
    Calculate the direction cosines for the center of a face, which are the angles
    made with the x, y, and z axes.
    """
    norm = np.linalg.norm(center)
    if norm == 0:
        # To handle the top and bottom face centers which are at the origin
        return (0, 0, 0)
    else:
        # Direction cosines are the normalized components of the center vector
        return tuple(np.degrees(np.arccos(coordinate/norm)) for coordinate in center)

# Calculate the direction cosines for all side faces
side_face_direction_cosines = [calculate_direction_cosines(center) for center in side_face_centers]

# Top and bottom faces are parallel or antiparallel to the z-axis, so their direction cosines are straightforward
top_face_direction_cosine = (90, 90, 0)  # Normal is pointing in the positive z-direction
bottom_face_direction_cosine = (90, 90, 180)  # Normal is pointing in the negative z-direction

# Combine the direction cosines for all faces
face_direction_cosines = side_face_direction_cosines + [top_face_direction_cosine, bottom_face_direction_cosine]

face_direction_cosines
