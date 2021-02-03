import os
import vtk
import tqdm
import numpy as np
from aicsimageio import AICSImage
from aicsshparam import shparam, shtools
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from skimage import segmentation as skseg
from skimage import morphology as skmorpho
from scipy import interpolate as spinterp
from scipy.ndimage import morphology as scimorph


def parameterization_from_shcoeffs(
    coeffs_mem,
    centroid_mem,
    coeffs_nuc,
    centroid_nuc,
    nisos=[32, 32],
    images_to_probe=None,
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

    representations = run_cellular_mapping(
        coeffs_mem=coeffs_mem,
        centroid_mem=centroid_mem,
        coeffs_nuc=coeffs_nuc,
        centroid_nuc=centroid_nuc,
        nisos=nisos,
        images_to_probe=images_to_probe,
    )

    return representations


def run_cellular_mapping(
    coeffs_mem, centroid_mem, coeffs_nuc, centroid_nuc, nisos, images_to_probe=None
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

    # Total number of coefficients
    nc = len(coeffs_mem)
    # Degree of the expansion (lmax)
    lmax = int(np.sqrt(nc / 2.0) - 1)

    if nc != len(coeffs_nuc):
        raise ValueError(
            "Number of coefficients in mem_coeffs and nuc_coeffs are "
            f"different: {nc,len(coeffs_nuc)}"
        )

    # Concatenate centroid into same array for interpolation
    centroids = np.c_[centroid_nuc, centroid_nuc, centroid_mem]

    # Array to be interpolated
    coeffs_ctr_arr = np.array([0 if i else 1 for i in range(nc)])
    coeffs_mem_arr = np.zeros((2, lmax + 1, lmax + 1))
    coeffs_nuc_arr = np.zeros((2, lmax + 1, lmax + 1))
    # Populate cell and nuclear arrays and concatenate into a single arr
    for k, kname in enumerate(["C", "S"]):
        for ll in range(lmax + 1):
            for m in range(lmax + 1):
                coeffs_mem_arr[k, ll, m] = coeffs_mem[f"L{ll}M{m}{kname}"]
                coeffs_nuc_arr[k, ll, m] = coeffs_nuc[f"L{ll}M{m}{kname}"]
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

    representations = []
    for i, iso_value in enumerate(np.linspace(0.0, 1.0, 1 + np.sum(nisos))):

        # Get coeffs at given fixed point
        coeffs = coeffs_interpolator(iso_value).reshape(2, lmax + 1, lmax + 1)
        mesh, grid = shtools.get_reconstruction_from_coeffs(coeffs, lrec=2 * lmax)

        # Translate mesh to interpolated location
        centroid = centroids_interpolator(iso_value).reshape(1, 3)
        mesh = translate_mesh(mesh, centroid)

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

    # Convert array into TIFF
    code = AICSImage(code)

    # Save channel names
    ch_names = []
    for ch_name, _ in representations[0].items():
        ch_names.append(ch_name)
    code.channel_names = ch_names

    return code


def get_intensity_representation(polydata, images_to_probe):

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


def get_croppped_version(mem, dna):

    z, y, x = np.where(mem)
    xmin = x.min() - 1
    xmax = x.max() + 2
    ymin = y.min() - 1
    ymax = y.max() + 2
    zmin = z.min() - 1
    zmax = z.max() + 2

    mem = mem[zmin:zmax, ymin:ymax, xmin:xmax]
    dna = dna[zmin:zmax, ymin:ymax, xmin:xmax]

    return mem, dna, (zmin, zmax, ymin, ymax, xmin, xmax)


def heat_eq_step(omega):

    """
    Executes a single step of the heat equation

    """

    omega[1:-1, 1:-1, 1:-1] += (
        omega[1:-1, 1:-1, :-2]
        + omega[1:-1, 1:-1, 2:]
        + omega[1:-1, :-2, 1:-1]
        + omega[1:-1, 2:, 1:-1]
        + omega[2:, 1:-1, 1:-1]
        + omega[:-2, 1:-1, 1:-1]
    ) / 6.0 - omega[1:-1, 1:-1, 1:-1]

    return omega


def get_geodesics(seg_mem, seg_nuc, nisos=8):

    """
    Calculates an approximation for geodesic distances within the cell
    by solving the heat equation at short time scales
    """

    geodesics = np.zeros_like(seg_mem)

    seg_mem, seg_nuc, roi = get_croppped_version(seg_mem, seg_nuc)

    # Create masks (1 = cytoplasm, 2 = nucleus)
    mask = (seg_mem > 0).astype(np.uint8) + (seg_nuc > 0).astype(np.uint8)

    # Calculates dmax
    edtm = scimorph.distance_transform_edt(mask > 0)
    dmax = int(edtm.max())

    # Values for Dirichlet boundary condition
    tmin = np.exp(0)
    tmid = np.exp(1)
    tmax = np.exp(2)

    # Create domain
    omega = tmid * np.ones(mask.shape, dtype=np.float32)

    # Find boundaries pixels
    warms = skseg.find_boundaries(mask > 0, connectivity=1, mode="inner")
    colds = skseg.find_boundaries(mask > 1, connectivity=1, mode="inner")
    freez = np.zeros_like(colds)
    z, y, x = np.where(mask == 2)
    freez[int(z.mean()), int(y.mean()), int(x.mean())] = True
    freez = skmorpho.binary_dilation(freez, selem=np.ones((13, 13, 13)))

    warms = np.where(warms)
    colds = np.where(colds)
    freez = np.where(freez)

    # Initial state
    omega[freez] = tmin
    omega[colds] = tmid
    omega[warms] = tmax

    # Solve heat equation for short time scale
    for run in range(50 * dmax):

        omega = heat_eq_step(omega)

        omega[freez] = tmin
        omega[colds] = tmid
        omega[warms] = tmax

    omega[mask == 0] = 0
    omega[omega > 0] = np.log(omega[omega > 0])

    # Estimate distances
    geodists = np.zeros_like(omega)
    # Exponentially spaced bins
    bins = np.exp(np.power(np.linspace(0.0, 1.0, nisos), 4))
    bins = 1 - (bins - bins.min()) / (bins.max() - bins.min())
    # Digitize values within the nucleus
    geodists[mask == 2] = 1 + np.digitize(omega[mask == 2].flatten(), bins[::-1])
    # Digitize values within the cytoplasm
    geodists[mask == 1] = (nisos + 1) + np.digitize(
        (omega[mask == 1] - 1).flatten(), bins[::-1]
    )
    # Set nuclear centroid as one
    geodists[int(z.mean()), int(y.mean()), int(x.mean())] = 1
    # Save
    geodists = geodists.astype(np.uint8)
    geodists[geodists == 0] = 1 + geodists.max()

    geodesics[roi[0] : roi[1], roi[2] : roi[3], roi[4] : roi[5]] = geodists

    geodesics[geodesics == 0] = geodists.max()

    return geodesics


def prob_image(polydata, images_to_probe):

    """
    Probe a multichannel image at the polydata points
    """

    n = polydata.GetNumberOfPoints()
    for img, name in images_to_probe:
        vmin = img.min()
        scalars = vtk.vtkFloatArray()
        scalars.SetNumberOfTuples(n)
        scalars.SetNumberOfComponents(1)
        scalars.SetName(name)
        for i in range(n):
            v = vmin
            r = polydata.GetPoint(i)
            try:
                v = img[int(r[2]), int(r[1]), int(r[0])]
            except Exception:
                pass
            scalars.SetTuple1(i, v)
        polydata.GetPointData().AddArray(scalars)

    return polydata


def get_geodesic_meshes(geodists, lmax=32, images_to_probe=None):

    """
    Convert image with geodesic distances into isosurfaces
    """

    iso_values = np.unique(geodists)[:-1]

    meshes = []
    for iso in iso_values:

        (coeffs, grid_rec), (image_, mesh, grid, transform) = shparam.get_shcoeffs(
            image=(geodists <= iso).astype(np.uint8), lmax=lmax, alignment_2d=False
        )

        centroid = (transform[0], transform[1], transform[2])

        mesh_rec = shtools.get_reconstruction_from_grid(grid_rec, centroid=centroid)

        if images_to_probe is not None:
            mesh_rec = prob_image(mesh_rec, images_to_probe)

        meshes.append(mesh_rec)

    return meshes


def interpolate_this_trace(trace, npts):

    """
    Interpolate a given trace
    """

    # length as the parametric variable
    length = np.cumsum(
        np.pad(np.sqrt(np.power(np.diff(trace, axis=0), 2).sum(axis=1)), (1, 0))
    )
    # normalize parametric variable from 0 to 1
    total_length = length[-1]
    length /= total_length
    # parametric trace
    trace_param = np.hstack([length.reshape(-1, 1), trace])
    # interpolate parametric trace
    ftrace = spinterp.interp1d(trace_param[:, 0].T, trace_param[:, 1:].T, kind="linear")
    trace_interp = ftrace(np.linspace(0, 1, npts)).T
    return trace_interp, total_length


def get_traces_(meshes, trace_pts=64, images_to_probe=None):

    pts = vtk.vtkPoints()
    cellarray = vtk.vtkCellArray()

    npts = meshes[0].GetNumberOfPoints()

    scalars = vtk.vtkFloatArray()
    scalars.SetNumberOfComponents(1)
    scalars.SetName("length")

    for i in range(npts):

        trace = []
        for mesh in meshes:
            r = mesh.GetPoint(i)
            trace.append(r)
        trace = np.array(trace)

        trace_interp, total_length = interpolate_this_trace(trace, npts=trace_pts)

        cellarray.InsertNextCell(len(trace_interp))

        for r in trace_interp:
            pid = pts.InsertNextPoint(r)
            cellarray.InsertCellPoint(pid)
            scalars.InsertNextTuple1(total_length)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(pts)
    polydata.SetLines(cellarray)
    polydata.GetPointData().AddArray(scalars)

    if images_to_probe is not None:
        polydata = prob_image(polydata, images_to_probe)

    return polydata


def get_traces(meshes, images_to_probe=None):

    pts = vtk.vtkPoints()
    cellarray = vtk.vtkCellArray()

    npts = meshes[0].GetNumberOfPoints()

    scalars_isoval = vtk.vtkFloatArray()
    scalars_isoval.SetNumberOfComponents(1)
    scalars_isoval.SetName("isoval")

    scalars_length = vtk.vtkFloatArray()
    scalars_length.SetNumberOfComponents(1)
    scalars_length.SetName("length")

    for i in range(npts):

        trace = []
        for mesh in meshes:
            r = mesh.GetPoint(i)
            trace.append(r)
        trace = np.array(trace)

        cellarray.InsertNextCell(len(trace))

        dr = 0.0
        ro = [0, 0, 0]
        for iso, r in enumerate(trace):
            pid = pts.InsertNextPoint(r)
            cellarray.InsertCellPoint(pid)
            scalars_isoval.InsertNextTuple1(iso)
            dr += np.sqrt(((np.array(ro) - np.array(r)) ** 2).sum())
            scalars_length.InsertNextTuple1(dr)
            ro = r

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(pts)
    polydata.SetLines(cellarray)
    polydata.GetPointData().AddArray(scalars_isoval)
    polydata.GetPointData().AddArray(scalars_length)

    if images_to_probe is not None:
        polydata = prob_image(polydata, images_to_probe)

    return polydata


def voxelize_mesh(imagedata, shape, mesh, origin):

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

    scalars = imgstenc.GetOutput().GetPointData().GetScalars()

    img = vtk_to_numpy(scalars).reshape(shape)

    return img


def voxelize_meshes(meshes):

    # 1st mesh is used as reference (it should be the bigger one)
    mesh = meshes[0]

    # Find dimensions of the image
    coords = []
    for i in range(mesh.GetNumberOfPoints()):
        r = mesh.GetPoints().GetPoint(i)
        coords.append(r)
    coords = np.array(coords)

    rmin = (coords.min(axis=0) - 0.5).astype(np.int)
    rmax = (coords.max(axis=0) + 0.5).astype(np.int)

    w = int(2 + (rmax[0] - rmin[0]))
    h = int(2 + (rmax[1] - rmin[1]))
    d = int(2 + (rmax[2] - rmin[2]))

    # Create image data
    imagedata = vtk.vtkImageData()
    imagedata.SetDimensions([w, h, d])
    imagedata.SetExtent(0, w - 1, 0, h - 1, 0, d - 1)
    imagedata.SetOrigin(rmin)
    imagedata.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

    for i in range(imagedata.GetNumberOfPoints()):
        imagedata.GetPointData().GetScalars().SetTuple1(i, 1)

    img = np.zeros((d, h, w), dtype=np.uint8)

    for mesh in meshes:
        seg = voxelize_mesh(
            imagedata=imagedata, shape=(d, h, w), mesh=mesh, origin=rmin
        )

        img += seg

    origin = rmin
    origin = origin.reshape(1, 3)

    return img, origin


def copy_content(sources, destination, arrays, normalize=False):

    data = {}
    for array in arrays:
        data[array] = []

    for source in tqdm.tqdm(sources):

        if os.path.exists(source):

            reader = vtk.vtkPolyDataReader()
            reader.SetFileName(source)
            reader.Update()

            mesh = reader.GetOutput()

            available_arrays = []
            for arr in range(mesh.GetPointData().GetNumberOfArrays()):
                available_arrays.append(mesh.GetPointData().GetArrayName(arr))

            for arr, array in enumerate(arrays):

                if array in available_arrays:

                    scalars = mesh.GetPointData().GetArray(array)

                    scalars = vtk_to_numpy(scalars)

                    data[array].append(scalars)

                else:

                    print(
                        f"WARNING: Array {array} not found."
                        f"Arrays available {available_arrays}. Source: {source}"
                    )

    npts = destination.GetNumberOfPoints()

    for array in arrays:

        for op, suffix in zip([np.mean, np.std], ["avg", "std"]):

            if len(data[array]) > 0:
                data_op = op(np.array(data[array]), axis=0)
            else:
                # If no signal found
                data_op = np.zeros(npts)

            new_array = vtk.vtkFloatArray()
            new_array.SetName(f"{array}_{suffix}")
            new_array.SetNumberOfComponents(1)
            new_array.SetNumberOfTuples(npts)
            for i in range(npts):
                new_array.SetTuple1(i, data_op[i])
            destination.GetPointData().AddArray(new_array)

    return destination


def translate_mesh(polydata, dr):

    if dr.shape != (1, 3):
        raise ValueError(
            f"Shape mismatch. dr has shape: {dr.shape}."
            "Expected: (1,3) for 3D displacement vector."
        )

    coords = vtk_to_numpy(polydata.GetPoints().GetData())

    coords += dr

    polydata.GetPoints().SetData(numpy_to_vtk(coords))

    return polydata


"""
def evaluate_reconstruction(seg_nuc, seg_mem, image_to_probe, array_name):

    # Run parameterization
    meshes, traces = parametrize(
        seg_mem=seg_mem, seg_nuc=seg_nuc, images_to_probe=[(image_to_probe, array_name)]
    )

    # Create projections (top view is right above the nucleus)
    z, y, _ = np.where(seg_nuc)
    zs = 1 + int(z.max())
    ys = int(y.mean())
    image_to_probe[seg_mem == 0] = 0

    projs = np.array(
        [
            np.vstack([seg_mem[zs], seg_mem[:, ys][::-1]]),
            np.vstack([seg_nuc[zs], seg_nuc[:, ys][::-1]]),
            np.vstack([image_to_probe[zs], image_to_probe[:, ys][::-1]]),
            np.vstack([seg_mem.max(axis=0), seg_mem.max(axis=1)[::-1]]),
            np.vstack([seg_nuc.max(axis=0), seg_nuc.max(axis=1)[::-1]]),
            np.vstack([image_to_probe.max(axis=0), image_to_probe.max(axis=1)[::-1]]),
        ]
    )

    # Voxelize result
    trc_coords = vtk_to_numpy(traces.GetPoints().GetData())
    trc_scalar = vtk_to_numpy(traces.GetPointData().GetArray(array_name))
    field = np.zeros(seg_mem.shape, dtype=np.float32)
    field[
        trc_coords[:, 2].astype(np.int),
        trc_coords[:, 1].astype(np.int),
        trc_coords[:, 0].astype(np.int),
    ] = trc_scalar
    field[seg_mem == 0] = 0

    # Fit
    nninterpolator = spinterp.NearestNDInterpolator(
        trc_coords[:, [2, 1, 0]], trc_scalar
    )

    # Interpolate
    cytopts = np.where(seg_mem)
    field_interp = field.copy()
    field_interp[cytopts] = nninterpolator(cytopts)

    # Projection of reconstructions
    projs_rec = np.array(
        [
            np.vstack([seg_mem[zs], seg_mem[:, ys][::-1]]),
            np.vstack([seg_nuc[zs], seg_nuc[:, ys][::-1]]),
            np.vstack([field[zs], field[:, ys][::-1]]),
            np.vstack([field_interp[zs], field_interp[:, ys][::-1]]),
            np.vstack([seg_mem.max(axis=0), seg_mem.max(axis=1)[::-1]]),
            np.vstack([seg_nuc.max(axis=0), seg_nuc.max(axis=1)[::-1]]),
            np.vstack([field.max(axis=0), field.max(axis=1)[::-1]]),
            np.vstack([field_interp.max(axis=0), field_interp.max(axis=1)[::-1]]),
        ]
    )

    # Pearson correlation between original and reconstruction
    valids = np.where(seg_mem)
    pcorr = np.corrcoef(image_to_probe[valids], field_interp[valids])[0, 1]

    return meshes, traces, (projs, projs_rec), pcorr
"""

if __name__ == "__main__":

    print("Helper functions for cytoparam.")
