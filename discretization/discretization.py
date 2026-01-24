import numpy as np
from typing import Tuple

from matplotlib import pyplot as plt

from mesh.mesh import Mesh


# TODO: function that checks if a point is inside the geometry (point in polygon test)
# TODO: then you can make a solid mask over the geometry



def find_mesh_square(
        mesh: Mesh,
        x_geom: float,
        y_geom: float
)-> Tuple[float, float, float, float]:
    """
    Function finds the mesh square in wich the point (x_geom, y_geom) lies and returns the
    mesh nodes.

    Parameters:
        mesh: (Mesh) target mesh.
        x_geom: (float) x coordinate of the geometry.
        y_geom: (float) y coordinate of the geometry.

    Returns:
        (Tuple[float, float, float, float]) the mesh nodes with (x_min, x_max, y_min, y_max).
    """
    x_1, x_2, y_1, y_2 = 0, 0, 0, 0
    if x_geom < mesh.x[0] or x_geom > mesh.x[-1]:
        raise ValueError(f'x_geom not in mesh. Mesh range: ({mesh.x[0], mesh.x[-1]}) given: {x_geom}.')
    if y_geom < mesh.y[0] or y_geom > mesh.y[-1]:
        raise ValueError(f'y_geom not in mesh. Mesh range: ({mesh.y[0], mesh.y[-1]}) given: {y_geom}.')

    for node_num, i in enumerate(mesh.x):
        if i > x_geom:
            x_1 = mesh.x[node_num - 1]
            x_2 = mesh.x[node_num]
            break

    for node_num, j in enumerate(mesh.y):
        if j > y_geom:
            y_1 = mesh.y[node_num - 1]
            y_2 = mesh.y[node_num]
            break

    return x_1, x_2, y_1, y_2

def find_nearest_mesh_node(
        mesh: Mesh,
        x_geom: float,
        y_geom: float
) -> Tuple[float, float]:
    """
    Function finds the nearest mesh node in the mesh to map to the wanted (x, y) coordinates of the
    geometry and returns it. If the point is in the middle of the two mesh nodes the smallest is chosen.

    Parameters:
        mesh: (Mesh) target mesh.
        x_geom: (float) x coordinate of the geometry.
        y_geom: (float) y coordinate of the geometry.

    Returns:
        (Tuple[float, float]) the nearest mesh point coordinates.
    """
    x_min, x_max, y_min, y_max = find_mesh_square(mesh, x_geom, y_geom)

    if abs(x_min - x_geom) > abs(x_max - x_geom):
        x_nearest = x_max
    else:
        x_nearest = x_min

    if abs(y_min - y_geom) > abs(y_max - y_geom):
        y_nearest = y_max
    else:
        y_nearest = y_min

    return x_nearest, y_nearest

class Geometry:
    def __init__(self, coordinates: np.ndarray):
        self.coordinates = coordinates

    def map_geometry_to_mesh(self, mesh: Mesh) -> None:

        for i, coordinate in enumerate(self.coordinates):
              self.coordinates[i, 0], self.coordinates[i, 1] = find_nearest_mesh_node(mesh, coordinate[0], coordinate[1])


if __name__ == '__main__':

    x_geom, y_geom, z_geom = np.loadtxt(r'X:\Python\geometry_discretization\geom.txt', delimiter='\t', unpack=True)

    geom_coordinates = np.column_stack([x_geom, y_geom])
    geom2_coordinates = np.column_stack([x_geom, y_geom])


    x = np.linspace(-0.5, 3.5, 10)
    y = np.linspace(-0.4, 1, 10)
    mesh = Mesh()
    mesh.create_mesh(x, y)

    geom_original = Geometry(geom_coordinates)
    geom_new = Geometry(geom2_coordinates)
    geom_new.map_geometry_to_mesh(mesh)

    fig, ax = plt.subplots()

    ax.plot(mesh.X, mesh.Y, linestyle='None', marker='o', color='black')


    ax.plot(geom_original.coordinates[:, 0], geom_original.coordinates[:, 1], linestyle='None', marker='x', color='red')
    ax.plot(geom_new.coordinates[:, 0], geom_new.coordinates[:, 1], linestyle='None', marker='x', color='blue')
    plt.axis("equal")
    plt.show()
