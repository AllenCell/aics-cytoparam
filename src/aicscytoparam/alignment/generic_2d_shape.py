import numpy as np
import matplotlib.pyplot as plt
from scipy import spatial as spspatial
from skimage import transform as sktrans
from skimage import measure as skmeasure


class Generic2DShape:
    """
    Generic class for 2D shapes
    """

    def __init__():
        pass

    def _compute_contour(self):
        """
        Compute the contour of the shape
        """
        cont = skmeasure.find_contours(self._polygon)[0]
        cx, cy = cont[:, 1], cont[:, 0]
        self.cx = cx - cx.mean()
        self.cy = cy - cy.mean()
        return

    def show(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.cx, self.cy)
        if ax is None:
            plt.show()

    def find_angle_that_minimizes_countour_distance(self, cx, cy):
        """
        Find the angle that minimizes the distance between the
        shape and the contour (cx, cy)
        Parameters
        ----------
        cx: np.ndarray
            x coordinates of the contour
        cy: np.ndarray
            y coordinates of the contour
        Returns
        -------
        angle: int
            angle that minimizes the distance
        dist: float
            minimum distance
        """
        # Assumes cx and cy are centered at origin
        dists = []
        for theta in range(180):
            cx2rot, cy2rot = Generic2DShape.rotate_contour(cx, cy, theta)
            X = np.c_[self.cx, self.cy]
            Y = np.c_[cx2rot, cy2rot]
            D = spspatial.distance.cdist(X, Y)
            dist_min = D.min(axis=0).mean() + D.min(axis=1).mean()
            dists.append(dist_min)
        return np.argmin(dists), np.min(dists)

    @staticmethod
    def rotate_contour(cx, cy, theta):
        """
        Rotate a contour around the origin
        Parameters
        ----------
        cx: np.ndarray
            x coordinates of the contour
        cy: np.ndarray
            y coordinates of the contour
        theta: float
            angle of rotation
        Returns
        -------
        cxrot: np.ndarray
            x coordinates of the rotated contour
        cyrot: np.ndarray
            y coordinates of the rotated contour
        """
        cxrot = cx * np.cos(np.deg2rad(theta)) - cy * np.sin(np.deg2rad(theta))
        cyrot = cx * np.sin(np.deg2rad(theta)) + cy * np.cos(np.deg2rad(theta))
        return cxrot, cyrot

    @staticmethod
    def get_contour_from_3d_image(image, pad=5, center=True):
        """
        Get the contour of a 3D image
        Parameters
        ----------
        image: np.ndarray
            3D image
        pad: int
            padding
        center: bool
            center the contour
        Returns
        -------
        cx: np.ndarray
            x coordinates of the contour
        cy: np.ndarray
            y coordinates of the contour
        """
        mip = image.max(axis=0)
        y, x = np.where(mip > 0)
        mip = np.pad(mip, ((pad, pad), (pad, pad)))
        cont = skmeasure.find_contours(mip > 0)[0]
        cx, cy = cont[:, 1], cont[:, 0]
        if center:
            cx = cx - cx.mean()
            cy = cy - cy.mean()
        return (cx, cy)


class ElongatedHexagonalShape(Generic2DShape):
    """
    Elongated hexagonal shape
    """

    def __init__(self, base, elongation, pad=5):
        self._pad = pad
        self._base = base
        self._height = int(self._base / np.sqrt(2))
        self._elongation_factor = elongation
        self._create()
        self._compute_contour()

    def _create(self):
        """
        Create the elongated hexagonal shape
        """
        pad = self._pad
        triangle = np.tril(np.ones((self._height, self._base)))
        triangle = sktrans.rotate(triangle, angle=-15, center=(0, 0), order=0)
        rectangle = np.ones((self._height, self._base))
        for _ in range(self._elongation_factor):
            rectangle = np.concatenate([rectangle, rectangle[:, :1]], axis=1)
        upper_half = np.concatenate([triangle[:, ::-1], rectangle, triangle], axis=1)
        hexagon = np.concatenate([upper_half, upper_half[::-1]], axis=0)
        hexagon = np.pad(hexagon, ((pad, pad), (pad, pad)))
        self._polygon = hexagon
        return

    @staticmethod
    def get_default_parameters_as_dict(elongation=8, base_ini=24, base_end=64):
        params = []
        for wid, w in enumerate(np.linspace(base_ini, base_end, elongation)):
            for fid, f in enumerate(np.linspace(0, w, elongation)):
                params.append({"base": int(w), "elongation": int(f)})
        return params
