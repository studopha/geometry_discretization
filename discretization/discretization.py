import numpy as np
from typing import Tuple, List

from matplotlib import pyplot as plt

from mesh.mesh import Mesh


# TODO: function that checks if a point is inside the geometry (point in polygon test)
# TODO: followed by a solid mask over the geometry

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

class OldDiscrete:
    def __init__(self, coordinates: np.ndarray):
        self.coordinates = coordinates




    def map_geometry_to_mesh(self, mesh: Mesh) -> None:

        for i, coordinate in enumerate(self.coordinates):
              self.coordinates[i, 0], self.coordinates[i, 1] = find_nearest_mesh_node(mesh, coordinate[0], coordinate[1])

class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

class Geometry:
    def __init__(self, surface_coordinates: np.ndarray):
        self.surface_coordinates = surface_coordinates

class DiscreteGeometry:
    def __init__(self, geometry: Geometry, mesh: Mesh):
        self.geometry = geometry
        self.mesh = mesh
        self.surface_nodes: List[Point] = []
        self._discretize_geometry()
        self._interpolate_missing_nodes()

    def _find_mesh_square(
            self,
            x_geom: float,
            y_geom: float
    ) -> Tuple[float, float, float, float]:
        x_1, x_2, y_1, y_2 = 0, 0, 0, 0

        if x_geom < self.mesh.x[0] or x_geom > self.mesh.x[-1]:
            raise ValueError(f'ERROR: x_geom not in mesh. Mesh range: ({self.mesh.x[0], self.mesh.x[-1]}) given: {x_geom}.')
        if y_geom < self.mesh.y[0] or y_geom > self.mesh.y[-1]:
            raise ValueError(f'ERROR: y_geom not in mesh. Mesh range: ({self.mesh.y[0], self.mesh.y[-1]}) given: {y_geom}.')

        for node_num, i in enumerate(self.mesh.x):
            if i > x_geom:
                x_1 = self.mesh.x[node_num - 1]
                x_2 = self.mesh.x[node_num]
                break

        for node_num, j in enumerate(self.mesh.y):
            if j > y_geom:
                y_1 = self.mesh.y[node_num - 1]
                y_2 = self.mesh.y[node_num]
                break

        return x_1, x_2, y_1, y_2

    def find_nearest_mesh_node(
            self,
            x_geom: float,
            y_geom: float
    )-> Tuple[float, float]:

        x_min, x_max, y_min, y_max = self._find_mesh_square(x_geom, y_geom)

        if abs(x_min - x_geom) > abs(x_max - x_geom):
            x_nearest = x_max
        else:
            x_nearest = x_min

        if abs(y_min - y_geom) > abs(y_max - y_geom):
            y_nearest = y_max
        else:
            y_nearest = y_min

        return x_nearest, y_nearest

    def _discretize_geometry(self) -> None:

        for i, coordinate in enumerate(self.geometry.surface_coordinates):
            x_node, y_node = self.find_nearest_mesh_node(coordinate[0], coordinate[1])
            node = Point(x_node, y_node)
            self.surface_nodes.append(node)

    def _coordinate_to_index(self, x: float, y: float) -> tuple[int, int]:

        ix = int(np.searchsorted(self.mesh.x, x))
        iy = int(np.searchsorted(self.mesh.y, y))

        ix = max(0, min(ix, len(self.mesh.x) - 1))
        iy = max(0, min(iy, len(self.mesh.y) - 1))

        if self.mesh.x[ix] != x:
            ix = int(np.argmin(np.abs(self.mesh.x - x)))
        if self.mesh.y[iy] != y:
            iy = int(np.argmin(np.abs(self.mesh.y - y)))

        return ix, iy

    def _index_to_coordinate(self, ix: int, iy: int) -> tuple[float, float]:
        return float(self.mesh.x[ix]), float(self.mesh.y[iy])

    def _bresenham_ij(self, i0: int, j0: int, i1: int, j1: int) -> list[tuple[int, int]]:
        """
        Bresenham line algorithm on integer grid indices (i,j).
        Returns all grid points from start to end (inclusive).
        """
        points: list[tuple[int, int]] = []

        di = abs(i1 - i0)
        dj = abs(j1 - j0)
        si = 1 if i0 < i1 else -1
        sj = 1 if j0 < j1 else -1

        err = di - dj

        i, j = i0, j0
        while True:
            points.append((i, j))
            if i == i1 and j == j1:
                break
            e2 = 2 * err
            if e2 > -dj:
                err -= dj
                i += si
            if e2 < di:
                err += di
                j += sj

        return points

    def _interpolate_missing_nodes(self) -> None:
        if not self.surface_nodes:
            return

        cleaned: list[Point] = [self.surface_nodes[0]]
        for point in self.surface_nodes[1:]:
            if point.x != cleaned[-1].x or point.y != cleaned[-1].y:
                cleaned.append(point)

        if len(cleaned) < 2:
            self.surface_nodes = cleaned
            return

        if cleaned[0].x == cleaned[-1].x and cleaned[0].y == cleaned[-1].y:
            cleaned.pop()

        n = len(cleaned)
        if n < 2:
            self.surface_nodes = cleaned
            return

        new_nodes: list[Point] = []
        seen_ij: set[tuple[int, int]] = set()

        for k in range(n):
            p0 = cleaned[k]
            p1 = cleaned[(k + 1) % n]

            i0, j0 = self._coordinate_to_index(p0.x, p0.y)
            i1, j1 = self._coordinate_to_index(p1.x, p1.y)

            ij_points = self._bresenham_ij(i0, j0, i1, j1)

            for (i, j) in ij_points:
                if (i, j) in seen_ij:
                    continue
                x, y = self._index_to_coordinate(i, j)
                new_nodes.append(Point(x, y))
                seen_ij.add((i, j))

        self.surface_nodes = new_nodes


if __name__ == '__main__':

    x_geom, y_geom, z_geom = np.loadtxt(r'X:\Python\geometry_discretization\geom.txt', delimiter='\t', unpack=True)

    geom_coordinates = np.column_stack([x_geom, y_geom])
    geom2_coordinates = np.column_stack([x_geom, y_geom])


    x = np.linspace(-0.5, 3.5, 1000)
    y = np.linspace(-0.4, 1, 1000)
    test_mesh = Mesh(x, y)

    print(f'dx: {test_mesh.x_dist}, dy: {test_mesh.y_dist}')

    geom_original = OldDiscrete(geom_coordinates)
    geom_new = OldDiscrete(geom2_coordinates)
    geom_new.map_geometry_to_mesh(test_mesh)

    test_geom = Geometry(geom_coordinates)
    test_discrete = DiscreteGeometry(test_geom, test_mesh)



    fig, ax = plt.subplots()

    #ax.plot(test_mesh.X, test_mesh.Y, linestyle='None', marker='o', color='black')

    for node in test_discrete.surface_nodes:
        ax.plot(node.x, node.y, linestyle='None', marker='x', color='red')
        1


    #ax.plot(test_discrete.geometry.surface_coordinates[:, 0], test_discrete.geometry.surface_coordinates[:, 1], linestyle='None', marker='x', color='blue')

    #ax.plot(geom_original.coordinates[:, 0], geom_original.coordinates[:, 1], linestyle='None', marker='x', color='blue')
    ax.plot(geom_new.coordinates[:, 0], geom_new.coordinates[:, 1], linestyle='None', marker='x', color='blue')
    plt.axis("equal")
    plt.show()
