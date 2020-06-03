import os
import vtk
import numpy as np
from aicsshparam import shparam, shtools
from vtk.util.numpy_support import vtk_to_numpy

from skimage import segmentation as skseg
from skimage import morphology as skmorpho
from scipy import interpolate as spinterp
from scipy.ndimage import morphology as scimorph

def get_croppped_version(mem, dna):

    z, y, x = np.where(mem)
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    zmin = z.min()
    zmax = z.max()

    mem = mem[zmin:zmax,ymin:ymax,xmin:xmax]
    dna = dna[zmin:zmax,ymin:ymax,xmin:xmax]

    return mem, dna, (zmin,zmax,ymin,ymax,xmin,xmax)

def heat_eq_step(omega):

    '''
        Executes a single step of the heat equation

        ALTERNATIVE:

        idx,idy,idz = np.unravel_index(id_1d,(3,3,3))
        np.ravel_multi_index((idx,idy,idz),(3,3,3))

    '''

    omega[1:-1, 1:-1, 1:-1] += (
            omega[1:-1, 1:-1,  :-2] +
            omega[1:-1, 1:-1,   2:] +
            omega[1:-1,  :-2, 1:-1] +
            omega[1:-1,   2:, 1:-1] +
            omega[  2:, 1:-1, 1:-1] +
            omega[ :-2, 1:-1, 1:-1]
        ) / 6. - omega[1:-1, 1:-1, 1:-1]
        
    return omega

def get_geodesics(seg_mem, seg_dna, nisos=8):

    '''
    Calculates an approximation for geodesic distances within the cell
    by solving the heat equation at short time scales
    '''

    geodesics = np.zeros_like(seg_mem)

    seg_mem, seg_dna, roi = get_croppped_version(seg_mem, seg_dna)

    # Create masks (1 = cytoplasm, 2 = nucleus)
    mask = (seg_mem>0).astype(np.uint8) + (seg_dna>0).astype(np.uint8)

    # Calculates dmax
    edtm = scimorph.distance_transform_edt(mask>0)
    dmax = int(edtm.max())

    # Values for Dirichlet boundary condition
    tmin = np.exp(0)
    tmid = np.exp(1)
    tmax = np.exp(2)

    # Create domain
    omega = tmid*np.ones(mask.shape, dtype=np.float64)

    # Find boundaries pixels
    warms = skseg.find_boundaries(mask>0, connectivity=1, mode='thick')
    colds = skseg.find_boundaries(mask>1, connectivity=1, mode='thick')
    freez = np.zeros_like(colds)
    z, y, x = np.where(mask==2)
    freez[int(z.mean()),int(y.mean()),int(x.mean())] = True
    freez = skmorpho.binary_dilation(freez, selem=np.ones((13,13,13)))

    warms = np.where(warms)
    colds = np.where(colds)
    freez = np.where(freez)

    # Initial state
    omega[freez] = tmin
    omega[colds] = tmid
    omega[warms] = tmax

    # Solve heat equation for short time scale
    for run in range(50*dmax):
        
        omega = heat_eq_step(omega)
                            
        omega[freez] = tmin
        omega[colds] = tmid
        omega[warms] = tmax

    omega[mask==0] = 0
    omega[omega>0] = np.log(omega[omega>0])

    # Estimate distances
    geodists = np.zeros_like(omega)
    # Exponentially spaced bins
    bins = np.exp(np.power(np.linspace(0.0,1.0,nisos), 4))
    bins = 1 - (bins-bins.min())/(bins.max()-bins.min())
    # Digitize values within the nucleus
    geodists[mask==2] = 1 + np.digitize(omega[mask==2].flatten(), bins[::-1])
    # Digitize values within the cytoplasm
    geodists[mask==1] = (nisos+1) + np.digitize((omega[mask==1]-1).flatten(), bins[::-1])
    # Set nuclear centroid as one
    geodists[int(z.mean()),int(y.mean()),int(x.mean())] = 1
    # Save
    geodists = geodists.astype(np.uint8)
    geodists[geodists==0] = 1 + geodists.max()

    geodesics[roi[0]:roi[1],roi[2]:roi[3],roi[4]:roi[5]] = geodists

    geodesics[geodesics==0] = geodists.max()

    return geodesics

def prob_image(polydata, images_to_probe):

    '''
    Probe a multichannel image at the polydata points
    '''

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
                v = img[int(r[2]),int(r[1]),int(r[0])]
            except: pass
            scalars.SetTuple1(i,v)
        polydata.GetPointData().AddArray(scalars)

    return polydata

def get_geodesic_meshes(geodists, lmax=32, alignment_mode=None, images_to_probe=None):

    '''
    Convert image with geodesic distances into isosurfaces
    '''

    iso_values = np.unique(geodists)[:-1]

    meshes = []
    for iso in iso_values:
        (coeffs, grid_rec), (image_, mesh, centroid, grid) = shparam.get_shcoeffs(
            image = (geodists<=iso).astype(np.uint8),
            lmax = lmax,
            alignment_mode = alignment_mode)
        mesh_rec = shtools.get_reconstruction_from_grid(grid_rec, centroid=centroid)
        if images_to_probe is not None:
            mesh_rec = prob_image(mesh_rec,images_to_probe)
        meshes.append(mesh_rec)

    return meshes

def interpolate_this_trace(trace, npts):

    '''
    Interpolate a given trace
    '''

    # length as the parametric variable
    length = np.cumsum(
        np.pad(np.sqrt(np.power(np.diff(trace, axis=0),2).sum(axis=1)), (1,0))
    )
    # normalize parametric variable from 0 to 1
    total_length = length[-1]
    length /= total_length
    # parametric trace
    trace_param = np.hstack([length.reshape(-1,1),trace])
    # interpolate parametric trace
    ftrace = spinterp.interp1d(trace_param[:,0].T, trace_param[:,1:].T, kind='linear')
    trace_interp = ftrace(np.linspace(0,1,npts)).T
    return trace_interp, total_length

def get_traces(meshes, trace_pts=64, images_to_probe=None):

    pts = vtk.vtkPoints()
    cellarray = vtk.vtkCellArray()

    npts = meshes[0].GetNumberOfPoints()

    scalars = vtk.vtkFloatArray()
    scalars.SetNumberOfComponents(1)
    scalars.SetName('length')

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
        polydata = prob_image(polydata,images_to_probe)

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

    rmin = (coords.min(axis=0)-0.5).astype(np.int)
    rmax = (coords.max(axis=0)+0.5).astype(np.int)

    w = int(2 + (rmax[0]-rmin[0]))
    h = int(2 + (rmax[1]-rmin[1]))
    d = int(2 + (rmax[2]-rmin[2]))

    # Create image data
    imagedata = vtk.vtkImageData()
    imagedata.SetDimensions([w,h,d])
    imagedata.SetExtent(0, w-1, 0, h-1, 0, d-1)
    imagedata.SetOrigin(rmin)
    imagedata.AllocateScalars(vtk.VTK_UNSIGNED_CHAR,1)

    for i in range(imagedata.GetNumberOfPoints()):
        imagedata.GetPointData().GetScalars().SetTuple1(i, 1)

    img = np.zeros((d,h,w), dtype=np.uint8)

    for mesh in meshes:
        seg = voxelize_mesh(
                    imagedata = imagedata,
                    shape = (d,h,w),
                    mesh = mesh,
                    origin = rmin
                )

        img += seg

    return img

def parametrize(seg_mem, seg_dna, images_to_probe=None):

    geodesics = get_geodesics(seg_mem=seg_mem, seg_dna=seg_mem)

    meshes = get_geodesic_meshes(
        geodists = geodesics,
        images_to_probe = images_to_probe
    )

    traces = get_traces(
        meshes = meshes,
        images_to_probe = images_to_probe
    )

    return meshes, traces

if __name__ == "__main__":

    print("Helper functions for cytoparam.")
