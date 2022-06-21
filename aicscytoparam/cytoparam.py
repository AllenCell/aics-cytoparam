import vtk
import warnings
import numpy as np
from aicsimageio import AICSImage
from aicsshparam import shparam, shtools
from scipy import interpolate as spinterp
from typing import Optional, List, Dict, Tuple
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk


def parameterize_image_coordinates(
    seg_mem: np.array, seg_nuc: np.array, lmax: int, nisos: List
):

    """
    Runs the parameterization for a cell represented by its spherical
    harmonics coefficients calculated by using ther package aics-shparam.

    Parameters
    --------------------
    seg_mem: np.array
        3D binary cell segmentation.
    seg_nuc: np.array
        3D binary nuclear segmentation.
    lmax: int
        Degree of spherical harmonics expansiion.
    nisos : list
        [a,b] representing the number of layers that will be used to
        parameterize the nucleoplasm and cytoplasm.

    Returns
    -------
    coords: np.array
        Array of shape 3xNxM, where NxM is the size of a
        parameterized intensity representation generated with
        same parameters lmax and nisos.
    coeffs_mem: dict
        Spherical harmonics coefficients that represent cell shape.
    centroid_mem: tuple
        (x,y,z) representing cell centroid.
    coeffs_nuc: dict
        Spherical harmonics coefficients that represent nuclear shape.
    centroid_nuc: tuple
        (x,y,z) representing nuclear centroid.
    """

    if (seg_mem.dtype != np.uint8) or (seg_nuc.dtype != np.uint8):
        warnings.warn(
            "One or more input images is not an 8-bit image\
        and will be cast to 8-bit."
        )

    # Cell SHE coefficients
    (coeffs_mem, _), (_, _, _, centroid_mem) = shparam.get_shcoeffs(
        image=seg_mem, lmax=lmax, sigma=0, compute_lcc=True, alignment_2d=False
    )

    # Nuclear SHE coefficients
    (coeffs_nuc, _), (_, _, _, centroid_nuc) = shparam.get_shcoeffs(
        image=seg_nuc, lmax=lmax, sigma=0, compute_lcc=True, alignment_2d=False
    )

    # Get Coordinates
    coords = get_mapping_coordinates(
        coeffs_mem=coeffs_mem,
        centroid_mem=centroid_mem,
        coeffs_nuc=coeffs_nuc,
        centroid_nuc=centroid_nuc,
        nisos=nisos,
    )

    # Shift coordinates to the center of the input
    # segmentations
    # coords += np.array(centroid_mem).reshape(3, 1, 1)

    return coords, (coeffs_mem, centroid_mem, coeffs_nuc, centroid_nuc)


def parameterization_from_shcoeffs(
    coeffs_mem: Dict,
    centroid_mem: List,
    coeffs_nuc: Dict,
    centroid_nuc: List,
    nisos: List,
    images_to_probe: Optional[List] = None,
):

    """
    Runs the parameterization for a cell represented by its spherical
    harmonics coefficients calculated by using ther package aics-shparam.

    Parameters
    --------------------
    coeffs_mem: dict
        Spherical harmonics coefficients that represent cell shape.
    centroid_mem: tuple
        (x,y,z) representing cell centroid
    coeffs_nuc: dict
        Spherical harmonics coefficients that represent nuclear shape.
    centroid_nuc: tuple
        (x,y,z) representing nuclear centroid
    nisos : list
        [a,b] representing the number of layers that will be used to
        parameterize the nucleoplasm and cytoplasm.
    images_to_probe : list of tuples
        [(a, b)] where a's are names for the image to be probed and b's are
        expected to be 3D ndarrays representing the image to be probed. The
        images are assumed to be previously registered to the cell and
        nuclear images used to calculate the spherical harmonics
        coefficients.

    Returns
    -------
    result: AICSImage
        Multidimensional image of shape 11C1YX, where Y is the number of
        interpolation levels and X is the number of points in the
        representation. C corresponds to the number of probed images.
    """

    if len(coeffs_mem) != len(coeffs_nuc):
        raise ValueError(
            f"Number of cell and nuclear coefficients\
        do not match: {len(coeffs_mem)} and {len(coeffs_nuc)}."
        )

    representations = cellular_mapping(
        coeffs_mem=coeffs_mem,
        centroid_mem=centroid_mem,
        coeffs_nuc=coeffs_nuc,
        centroid_nuc=centroid_nuc,
        nisos=nisos,
        images_to_probe=images_to_probe,
    )

    return representations


def get_interpolators(
    coeffs_mem: Dict,
    centroid_mem: Dict,
    coeffs_nuc: Dict,
    centroid_nuc: Dict,
    nisos: List,
):

    """
    Creates 1D interpolators for SHE coefficients with fixed points
    at: 1) nuclear centroid, 2) nuclear shell and 3) cell membrane.
    Also creates an interpolator for corresponding centroids.

    Parameters
    --------------------
    coeffs_mem: dict
        Spherical harmonics coefficients that represent cell shape.
    centroid_mem: tuple
        (x,y,z) representing cell centroid
    coeffs_nuc: dict
        Spherical harmonics coefficients that represent nuclear shape.
    centroid_nuc: tuple
        (x,y,z) representing nuclear centroid
    nisos : list
        [a,b] representing the number of layers that will be used to
        parameterize the nucleoplasm and cytoplasm.

    Returns
    -------
        coeffs_interpolator: spinterp.interp1d
        centroids_interpolator: spinterp.interp1d
        lmax: int
    """

    if len(coeffs_mem) != len(coeffs_nuc):
        raise ValueError(
            f"Number of cell and nuclear coefficients\
        do not match: {len(coeffs_mem)} and {len(coeffs_nuc)}."
        )

    # Total number of coefficients
    nc = len(coeffs_mem)
    # Degree of the expansion (lmax)
    lmax = int(np.sqrt(nc / 2.0) - 1)

    if nc != len(coeffs_nuc):
        raise ValueError(
            f"Number of coefficients in mem_coeffs and\
        nuc_coeffs are different:{nc,len(coeffs_nuc)}"
        )

    # Concatenate centroid into same array for interpolation
    centroids = np.c_[centroid_nuc, centroid_nuc, centroid_mem]

    # Array to be interpolated
    coeffs_ctr_arr = np.array([0 if i else 1 for i in range(nc)])
    coeffs_mem_arr = np.zeros((2, lmax + 1, lmax + 1))
    coeffs_nuc_arr = np.zeros((2, lmax + 1, lmax + 1))
    # Populate cell and nuclear arrays and concatenate into a single arr
    for k, kname in enumerate(["C", "S"]):
        for L in range(lmax + 1):
            for m in range(lmax + 1):
                coeffs_mem_arr[k, L, m] = coeffs_mem[f"shcoeffs_L{L}M{m}{kname}"]
                coeffs_nuc_arr[k, L, m] = coeffs_nuc[f"shcoeffs_L{L}M{m}{kname}"]
    coeffs_mem_arr = coeffs_mem_arr.flatten()
    coeffs_nuc_arr = coeffs_nuc_arr.flatten()
    coeffs = np.c_[coeffs_ctr_arr, coeffs_nuc_arr, coeffs_mem_arr]

    # Calculate fixed points for interpolation
    iso_values = [0.0] + nisos
    iso_values = np.cumsum(iso_values)
    iso_values = iso_values / iso_values[-1]

    # Coeffs interpolator
    coeffs_interpolator = spinterp.interp1d(iso_values, coeffs)

    # Centroid interpolator
    centroids_interpolator = spinterp.interp1d(iso_values, centroids)

    return coeffs_interpolator, centroids_interpolator, lmax


def get_mapping_coordinates(
    coeffs_mem: Dict,
    centroid_mem: List,
    coeffs_nuc: Dict,
    centroid_nuc: List,
    nisos: List,
):

    """
    Interpolate spherical harmonics coefficients representing the nuclear centroid,
    the nuclear shell and the cell membrane. As the coefficients are interpolated,
    polygonal meshes representing corresponding 3D shapes are reconstructed and
    the coordinates of their points are returned.

    Parameters
    --------------------
    coeffs_mem: dict
        Spherical harmonics coefficients that represent cell shape.
    centroid_mem: tuple
        (x,y,z) representing cell centroid
    coeffs_nuc: dict
        Spherical harmonics coefficients that represent nuclear shape.
    centroid_nuc: tuple
        (x,y,z) representing nuclear centroid
    nisos : list
        [a,b] representing the number of layers that will be used to
        parameterize the nucleoplasm and cytoplasm.

    Returns
    -------
    coords: np.array
        Array of shape 3xNxM, where NxM is the size of a
        parameterized intensity representation generated with
        same parameters nisos and same SHE degree.
    """

    if len(coeffs_mem) != len(coeffs_nuc):
        raise ValueError(
            f"Number of cell and nuclear coefficients do not\
        match: {len(coeffs_mem)} and {len(coeffs_nuc)}."
        )

    # Get interpolators
    coeffs_interpolator, centroids_interpolator, lmax = get_interpolators(
        coeffs_mem=coeffs_mem,
        centroid_mem=centroid_mem,
        coeffs_nuc=coeffs_nuc,
        centroid_nuc=centroid_nuc,
        nisos=nisos,
    )

    x_coords, y_coords, z_coords = [], [], []
    for i, iso_value in enumerate(np.linspace(0.0, 1.0, 1 + np.sum(nisos))):

        # Get coeffs at given fixed point
        coeffs = coeffs_interpolator(iso_value).reshape(2, lmax + 1, lmax + 1)
        mesh, _ = shtools.get_reconstruction_from_coeffs(coeffs, lrec=2 * lmax)

        # Store coordinates
        coords = vtk_to_numpy(mesh.GetPoints().GetData())

        coords += centroids_interpolator(iso_value).reshape(1, -1)

        x_coords.append(coords[:, 0])
        y_coords.append(coords[:, 1])
        z_coords.append(coords[:, 2])

    coords = np.array((x_coords, y_coords, z_coords))

    return coords


def cellular_mapping(
    coeffs_mem: Dict,
    centroid_mem: List,
    coeffs_nuc: Dict,
    centroid_nuc: List,
    nisos: List,
    images_to_probe: Optional[List] = None,
):

    """
    Interpolate spherical harmonics coefficients representing the nuclear centroid,
    the nuclear shell and the cell membrane. As the coefficients are interpolated,
    polygonal meshes representing corresponding 3D shapes are reconstructed and used
    to probe input images and create intensity representations.

    Parameters
    --------------------
    coeffs_mem: dict
        Spherical harmonics coefficients that represent cell shape.
    centroid_mem: tuple
        (x,y,z) representing cell centroid
    coeffs_nuc: dict
        Spherical harmonics coefficients that represent nuclear shape.
    centroid_nuc: tuple
        (x,y,z) representing nuclear centroid
    nisos : list
        [a,b] representing the number of layers that will be used to
        parameterize the nucleoplasm and cytoplasm.
    images_to_probe : list of tuples
        [(a, b)] where a's are names for the image to be probed and b's are
        expected to be 3D ndarrays representing the image to be probed. The
        images are assumed to be previously registered to the cell and
        nuclear images used to calculate the spherical harmonics
        coefficients.

    Returns
    -------
    result: AICSImage
        Multidimensional image of shape 11C1YX, where Y is the number of
        interpolation levels and X is the number of points in the
        representation. C corresponds to the number of probed images.
    """

    if len(coeffs_mem) != len(coeffs_nuc):
        raise ValueError(
            f"Number of cell and nuclear coefficients do not\
        match: {len(coeffs_mem)} and {len(coeffs_nuc)}."
        )

    # Get interpolators
    coeffs_interpolator, centroids_interpolator, lmax = get_interpolators(
        coeffs_mem=coeffs_mem,
        centroid_mem=centroid_mem,
        coeffs_nuc=coeffs_nuc,
        centroid_nuc=centroid_nuc,
        nisos=nisos,
    )

    representations = []
    for i, iso_value in enumerate(np.linspace(0.0, 1.0, 1 + np.sum(nisos))):

        # Get coeffs at given fixed point
        coeffs = coeffs_interpolator(iso_value).reshape(2, lmax + 1, lmax + 1)
        mesh, grid = shtools.get_reconstruction_from_coeffs(coeffs, lrec=2 * lmax)

        # Translate mesh to interpolated location
        centroid = centroids_interpolator(iso_value).reshape(1, 3)
        coords = vtk_to_numpy(mesh.GetPoints().GetData())
        coords += centroid
        mesh.GetPoints().SetData(numpy_to_vtk(coords))

        # Probe images of interest to create representation
        rep = get_intensity_representation(
            polydata=mesh, images_to_probe=images_to_probe
        )

        representations.append(rep)

    # Number of points in the mesh
    npts = mesh.GetNumberOfPoints()

    # Create a matrix to store the rpresentations
    code = np.zeros(
        (len(images_to_probe), 1, len(representations), npts), dtype=np.float32
    )

    for i, rep in enumerate(representations):
        for ch, (ch_name, data) in enumerate(rep.items()):
            code[ch, 0, i, :] = data

    # Save channel names
    ch_names = []
    for ch_name, _ in representations[0].items():
        ch_names.append(ch_name)

    # Convert array into TIFF
    code = AICSImage(code, channel_names=ch_names)

    return code


def get_intensity_representation(polydata: vtk.vtkPolyData, images_to_probe: List):

    """
    This function probes the location of 3D mesh points in a list
    of 3D images.

    Parameters
    --------------------
    polydata: vtkPolyData
        Polygonal 3D mesh
    images_to_probe : list of tuples
        [(a, b)] where a's are names for the image to be probed and b's are
        expected to be 3D ndarrays representing the image to be probed. The
        images are assumed to be previously registered to the cell and
        nuclear images used to calculate the spherical harmonics
        coefficients.

    Returns
    -------
    result: list of ndarrays
        [(a,b)] where a is the probed image name and b is a 1d array with
        the corresponding probed intensities.
    """

    representation = {}
    coords = vtk_to_numpy(polydata.GetPoints().GetData())
    x, y, z = [coords[:, i].astype(np.int) for i in range(3)]
    for name, img in images_to_probe:
        # Bound the values of x, y and z coordinates to fit inside the
        # probe image
        x_clip = np.clip(x, 0, img.shape[2] - 1)
        y_clip = np.clip(y, 0, img.shape[1] - 1)
        z_clip = np.clip(z, 0, img.shape[0] - 1)
        representation[name] = img[z_clip, y_clip, x_clip]
    return representation


def voxelize_mesh(
    imagedata: vtk.vtkImageData, shape: Tuple, mesh: vtk.vtkPolyData, origin: List
):

    """
    Voxelize a triangle mesh into an image.

    Parameters
    --------------------
    imagedata: vtkImageData
        Imagedata that will be uses as support for voxelization.
    shape: tuple
        Shape that imagedata scalars will take after
        voxelization.
    mesh: vtkPolyData
        Mesh to be voxelized
    origin: List
        xyz specifying the lower left corner of the mesh.

    Returns
    -------
    img: np.array
        Binary array.
    """

    pol2stenc = vtk.vtkPolyDataToImageStencil()
    pol2stenc.SetInputData(mesh)
    pol2stenc.SetOutputOrigin(origin)
    pol2stenc.SetOutputWholeExtent(imagedata.GetExtent())
    pol2stenc.Update()

    imgstenc = vtk.vtkImageStencil()
    imgstenc.SetInputData(imagedata)
    imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
    imgstenc.ReverseStencilOff()
    imgstenc.SetBackgroundValue(0)
    imgstenc.Update()

    # Convert scalars from vtkImageData back to numpy
    scalars = imgstenc.GetOutput().GetPointData().GetScalars()
    img = vtk_to_numpy(scalars).reshape(shape)

    return img


def voxelize_meshes(meshes: List):

    """
    List of meshes to be voxelized into an image. Usually
    the input corresponds to the cell membrane and nuclear
    shell meshes.

    Parameters
    --------------------
    meshes: List
        List of vtkPolydatas representing the meshes to
        be voxelized into an image.
    Returns
    -------
    img: np.array
        3D image where voxels with value i represent are
        those found in the interior of the i-th mesh in
        the input list. If a voxel is interior to one or
        more meshes form the input list, it will take the
        value of the right most mesh in the list.
    origin:
        Origin of the meshes in the voxelized image.
    """

    # 1st mesh is used as reference (cell) and it should be
    # the larger than the 2nd one (nucleus).
    mesh = meshes[0]

    # Find mesh coordinates
    coords = vtk_to_numpy(mesh.GetPoints().GetData())

    # Find bounds of the mesh
    rmin = (coords.min(axis=0) - 0.5).astype(np.int)
    rmax = (coords.max(axis=0) + 0.5).astype(np.int)

    # Width, height and depth
    w = int(2 + (rmax[0] - rmin[0]))
    h = int(2 + (rmax[1] - rmin[1]))
    d = int(2 + (rmax[2] - rmin[2]))

    # Create image data
    imagedata = vtk.vtkImageData()
    imagedata.SetDimensions([w, h, d])
    imagedata.SetExtent(0, w - 1, 0, h - 1, 0, d - 1)
    imagedata.SetOrigin(rmin)
    imagedata.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

    # Set all values to 1
    imagedata.GetPointData().GetScalars().FillComponent(0, 1)

    # Create an empty 3D numpy array to sum up
    # voxelization of all meshes
    img = np.zeros((d, h, w), dtype=np.uint8)

    # Voxelize one mesh at the time
    for mid, mesh in enumerate(meshes):
        seg = voxelize_mesh(
            imagedata=imagedata, shape=(d, h, w), mesh=mesh, origin=rmin
        )
        img[seg > 0] = mid + 1

    # Origin of the reference system in the image
    origin = rmin.reshape(1, 3)

    return img, origin


def morph_representation_on_shape(
    img: np.array, param_img_coords: np.array, representation: np.array
):

    """
    Decodes the parameterized intensity representation
    into an image. To do so, an input image with the final
    shape is required.

    Parameters
    --------------------
    img: np.array
        3D binary image with the shape in which the
        representation will be decoded on.
    img_param_coords: np.array
        Array of shape 3NM, where N is the number of
    isovalues in the representation and M is the number of
    points in each mesh used to create the representation.
    This array should be generated by the function
    param_img_coords.
    representation: np.array
        Array of shape NM as generated by the function
        run_cellular_mapping.
    Returns
    -------
    img: np.array
        3D image where voxels have their values interpolated
        from the decoded representation
    """

    if param_img_coords.shape[-2:] != representation.shape[-2:]:
        raise ValueError(
            f"Parameterized image coordinates of\
        shape {param_img_coords.shape} and representation of\
        shape {representation.shape} are expected to have the\
        same shape."
        )

    # Reshape coords from 3NM to 3P, where N is the number of
    # isovalues in the representation and M is the number of
    # points in each mesh used to create the representation.
    param_img_coords = param_img_coords.reshape(3, -1).T
    # Convert XYZ to ZYX
    param_img_coords = param_img_coords[:, ::-1]
    # Reshape representation from NM to P.
    representation = representation.flatten()

    # Create a nearest neighbor interpolator
    nninterpolator = spinterp.NearestNDInterpolator(param_img_coords, representation)
    # Interpolate on all foreground voxels
    cell = np.where(img > 0)
    img = img.astype(np.float32)
    img[cell] = nninterpolator(cell)

    return img
