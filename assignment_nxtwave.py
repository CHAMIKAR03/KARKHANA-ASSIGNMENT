

import numpy as np
from scipy.integrate import dblquad
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class MobiusStrip:
    def __init__(self, R=1.0, w=0.3, n=200):
        self.R = R          # Radius from center to strip
        self.w = w          # Width of the strip
        self.n = n          # Resolution of the mesh
        self.u = np.linspace(0, 2 * np.pi, n)
        self.v = np.linspace(-w / 2, w / 2, n)
        self.U, self.V = np.meshgrid(self.u, self.v)
        self.X, self.Y, self.Z = self._generate_mesh()

    def _generate_mesh(self):
        """
        Uses the Mobius strip parametric equations to compute the mesh grid.
        """
        u, v, R = self.U, self.V, self.R
        x = (R + v * np.cos(u / 2)) * np.cos(u)
        y = (R + v * np.cos(u / 2)) * np.sin(u)
        z = v * np.sin(u / 2)
        return x, y, z

    def surface_area(self):
        """
        Approximates the surface area using numerical integration of the magnitude
        of the cross product of partial derivatives.
        """
        def integrand(u, v):
            R = self.R
            dxdu = - (R + v * np.cos(u / 2)) * np.sin(u) - 0.5 * v * np.sin(u / 2) * np.cos(u)
            dydu = (R + v * np.cos(u / 2)) * np.cos(u) - 0.5 * v * np.sin(u / 2) * np.sin(u)
            dzdu = 0.5 * v * np.cos(u / 2)

            dxdv = np.cos(u / 2) * np.cos(u)
            dydv = np.cos(u / 2) * np.sin(u)
            dzdv = np.sin(u / 2)

            # Cross product of partial derivatives
            normal = np.array([
                dydu * dzdv - dzdu * dydv,
                dzdu * dxdv - dxdu * dzdv,
                dxdu * dydv - dydu * dxdv
            ])
            return np.linalg.norm(normal)

        area, _ = dblquad(integrand, -self.w / 2, self.w / 2, lambda v: 0, lambda v: 2 * np.pi)
        return area

    def edge_length(self):
        """
        Approximates the edge length along both edges of the strip.
        """
        def get_edge_points(v_val):
            u = self.u
            x = (self.R + v_val * np.cos(u / 2)) * np.cos(u)
            y = (self.R + v_val * np.cos(u / 2)) * np.sin(u)
            z = v_val * np.sin(u / 2)
            return np.stack((x, y, z), axis=1)

        def compute_length(points):
            diffs = np.diff(points, axis=0)
            distances = np.linalg.norm(diffs, axis=1)
            return np.sum(distances)

        edge_top = get_edge_points(self.w / 2)
        edge_bottom = get_edge_points(-self.w / 2)
        return compute_length(edge_top) + compute_length(edge_bottom)

    def plot(self):
        """
        Plots the Mobius strip in 3D.
        """
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(self.X, self.Y, self.Z, rstride=4, cstride=4,
                        color='skyblue', edgecolor='k', alpha=0.9)
        ax.set_title("Mobius Strip")
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    mobius = MobiusStrip(R=1.0, w=0.3, n=300)
    area = mobius.surface_area()
    edge_len = mobius.edge_length()

    print(f"Approximate Surface Area: {area:.4f}")
    print(f"Approximate Edge Length: {edge_len:.4f}")

    # Show 3D plot
    mobius.plot()
