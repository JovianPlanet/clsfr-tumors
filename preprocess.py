import nibabel as nib
import nibabel.processing
import numpy as np

def preprocess(path, config):

    scan = nib.load(path)
    aff  = scan.affine
    vol  = np.int16(scan.get_fdata())

    # Resamplea volumen y affine a un nuevo shape
    new_zooms  = np.array(scan.header.get_zooms()) * config['new_z']
    new_shape  = np.array(vol.shape) // config['new_z']
    new_affine = nibabel.affines.rescale_affine(aff, vol.shape, new_zooms, new_shape)
    scan       = nibabel.processing.conform(scan, new_shape, new_zooms)
    ni_img     = nib.Nifti1Image(scan.get_fdata(), new_affine)
    vol        = np.int16(ni_img.get_fdata())

    return vol
