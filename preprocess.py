import nibabel as nib
import nibabel.processing
import numpy as np

def preprocess(path, config):

    scan = nib.load(path)
    scan = nibabel.processing.conform(scan, config['model_dims'])
    vol  = np.float32(scan.get_fdata())

    return vol
