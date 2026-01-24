import numpy as np

from typing import Tuple

from matplotlib import pyplot as plt



class Mesh:
    def __init__(self):
        self.x = None
        self.y = None
        self.X = None
        self.Y = None

    def create_mesh(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x = x
        self.y = y

        self.X, self.Y = np.meshgrid(self.x, self.y)

    def print_mesh(self, x_range: Tuple[int, int], y_range: Tuple[int, int]) -> None:

        for j in range(y_range[0], y_range[1]):
            print('\n')
            for i in range(x_range[0], x_range[1]):
                print(f"({self.X[j, i]:.3f}, {self.Y[j, i]:.3f})", end=" ")

    def plot_mesh(self, x_range: Tuple[int, int], y_range: Tuple[int, int]) -> None:
        fig, ax = plt.subplots()

        ax.plot(self.X[x_range[0]:x_range[1], y_range[0]:y_range[1]], linestyle='None', marker='o', color='black')

        plt.grid(True)
        plt.show()




if __name__ == "__main__":

    point = (2.5, 2.3)


    x = np.linspace(0, 100, 100)
    y = np.linspace(0, 100, 100)
    mesh = Mesh()
    mesh.create_mesh(x, y)
    mesh.print_mesh((0, 5), (0, 5))
    mesh.plot_mesh((0, 5), (0, 5), point)