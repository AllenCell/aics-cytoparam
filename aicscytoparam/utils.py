import numpy as np
import pandas as pd
from typing import List
from skimage import io as skio
from aicsshparam import shtools
from vtk.util import numpy_support as vtknp
from aicscytoparam.alignment.generic_2d_shape import Generic2DShape


def load_image_and_align_based_on_shape_library(path, library, nuc_channel=1):
    """
    Load an image and align it based on a shape library
    Parameters
    ----------
    path: str
        path to the image
    library: ShapeLibrary2D
        shape library
    nuc_channel: int
        channel of the nucleus
    Returns
    -------
    image_rot: np.ndarray
        aligned image
    """
    image = skio.imread(path)
    (cx, cy) = Generic2DShape.get_contour_from_3d_image(image[0])
    _, _, angle = library.find_best_match(cx, cy)
    image_rot = shtools.rotate_image_2d(image, -angle)
    # Apply extra 180 degrees rotation if nucleus is at the right
    # side of the aligned cell
    x = np.arange(image_rot.shape[-1])
    px = image_rot[nuc_channel].sum(axis=(0, 1))
    xpx = (x - x.mean()) * px / px.sum()
    if xpx.sum() > 0:
        image_rot = shtools.rotate_image_2d(image_rot, 180)
    return image_rot


def get_voxelized_image_of_mean_shape(
    vec: List, coeffs_mem: List, coeffs_nuc: List, return_meshes=False
):
    """
    Get the voxelized image of the mean shape
    Parameters
    ----------
    vec: List
        list of displacement vectors of nuclear centroid relative to cell centroid
    coeffs_mem: List
        list of membrane coefficients
    coeffs_nuc: List
        list of nuclear coefficients
    return_meshes: bool
        return the meshes
    Returns
    -------
    avg_image: np.ndarray
        voxelized image of the mean shape
    """
    vec = np.array(vec).mean(axis=0)
    df_coeffs_mem = pd.DataFrame(coeffs_mem)
    df_coeffs_nuc = pd.DataFrame(coeffs_nuc)
    df_coeffs_mem_avg = df_coeffs_mem.select_dtypes(include="number").mean(axis=0)
    df_coeffs_nuc_avg = df_coeffs_nuc.select_dtypes(include="number").mean(axis=0)
    avg_coeffs_mem = shtools.convert_coeffs_dict_to_matrix(df_coeffs_mem_avg, lmax=32)
    avg_coeffs_nuc = shtools.convert_coeffs_dict_to_matrix(df_coeffs_nuc_avg, lmax=32)
    avg_mesh_mem, _ = shtools.get_reconstruction_from_coeffs(avg_coeffs_mem)
    avg_mesh_nuc, _ = shtools.get_reconstruction_from_coeffs(avg_coeffs_nuc)
    # Translate nucleus to correct location based on average displacement of nuclear
    # centroid relative to cell centroid
    pts = vtknp.vtk_to_numpy(avg_mesh_nuc.GetPoints().GetData())
    shtools.update_mesh_points(
        avg_mesh_nuc, pts[:, 0] - vec[0], pts[:, 1] - vec[1], pts[:, 2] - vec[2]
    )
    # Convert average cell shape into an image
    avg_image, _ = shtools.voxelize_meshes([avg_mesh_mem, avg_mesh_nuc])
    if return_meshes:
        return avg_image, avg_mesh_mem, avg_mesh_nuc
    return avg_image
