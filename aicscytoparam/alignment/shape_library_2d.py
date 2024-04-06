import numpy as np
import matplotlib.pyplot as plt
from aicscytoparam.alignment.generic_2d_shape import Generic2DShape


class ShapeLibrary2D:
    """
    Define a library of 2D shapes
    """

    def __init__(self):
        pass

    def set_base_shape(self, polygon):
        """
        Set the base shape for the library
        Parameters
        ----------
        polygon: Generic2DShape
            base shape for the library
        """
        self._polygon = polygon

    def set_parameters_range(self, params_dict):
        """
        Set the parameters range for the library
        Parameters
        ----------
        params_dict: dict
            dictionary with the parameters range
        """
        self._params = params_dict

    def find_best_match(self, cx, cy):
        """
        Find the best match between the contour (cx, cy) and the shapes in the library
        Parameters
        ----------
        cx: np.ndarray
            x coordinates of the contour
        cy: np.ndarray
            y coordinates of the contour
        Returns
        -------
        idx: int
            index of the best match
        params: dict
            parameters of the best match
        angle: float
            angle that minimizes the distance
        """
        angles, dists = [], []
        for p in self._params:
            poly = self._polygon(**p)
            a, d = poly.find_angle_that_minimizes_countour_distance(cx, cy)
            angles.append(a)
            dists.append(d)
        idx = np.argmin(dists)
        return idx, self._params[idx], angles[idx]

    def display(self, xlim=[-150, 150], ylim=[-50, 50], contours_to_match=None):
        """
        Display the shapes in the library
        Parameters
        ----------
        xlim: list
            x limits of the plot
        ylim: list
            y limits of the plot
        contours_to_match: list of tuples
            list of tuples with the contours to match
        """
        n = int(np.sqrt(len(self._params)))
        fig, axs = plt.subplots(n, n, figsize=(3 * n, 1 * n))
        for pid, p in enumerate(self._params):
            j, i = pid // n, pid % n
            poly = self._polygon(**p)
            axs[j, i].plot(poly.cx, poly.cy, lw=7, color="k", alpha=0.2)
            axs[j, i].axis("off")
            axs[j, i].set_aspect("equal")
            axs[j, i].set_xlim(xlim[0], xlim[1])
            axs[j, i].set_ylim(ylim[0], ylim[1])
        if contours_to_match is not None:
            for cx, cy in contours_to_match:
                pid, p, angle = self.find_best_match(cx, cy)
                cxrot, cyrot = Generic2DShape.rotate_contour(cx, cy, angle)
                axs[j, i].plot(cxrot, cyrot, color="magenta")
        plt.tight_layout()
        plt.show()
