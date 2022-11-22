import os
import numpy as np
from tensorflow import keras
from skimage import transform

import nibabel as nib
import nibabel.processing

model_name = 'tumor_clf.h5'

model_path = os.path.join('/media',
                          'davidjm',
                          'Disco_Compartido',
                          'david',
                          'clsfr-tumors',
                          model_name
)

model = keras.models.load_model(model_path)
print(f'Model summary = {model.summary()}')
print(f'Model layers = {model.layers}')

ds = 'IATM-controls'

print(f'\nPredicciones para el {ds} dataset:\n')


if ds == 'IATM-controls':

    # imagen iatm control con resample
    reg_iatm_ctrl_path = os.path.join('/home',
                           'davidjm',
                           'Downloads',
                           'Reg-IATM',
                           'sujetos_proyecto_controles'
    )

    subjects = next(os.walk(reg_iatm_ctrl_path))[1]
    for subject in subjects:
        print(f'sujeto = {subject[:-4]}')
        studies = next(os.walk(os.path.join(reg_iatm_ctrl_path, subject)))[1]
        print(f'study = {studies}')
        p = os.path.join(reg_iatm_ctrl_path, subject, studies[0], 'NIFTI', 'sub-'+subject[:-4]+'_T1w.nii.gz')
        try:
            control = nib.load(p)
        except:
            print(f'No se encontro MRI para este estudio/paciente')
            continue
        control = nibabel.processing.conform(control, (128, 128, 64))
        control = control.get_fdata()
        control = np.expand_dims(control, 0)
        print(f'{model.predict(control)}')

elif ds == 'brats':
    brats_path = os.path.join('/home',
                              'davidjm',
                              'Downloads',
                              'BraTS-dataset',
                              'BraTS2020_ValidationData',
                              'MICCAI_BraTS2020_ValidationData'
    )

    subjects = next(os.walk(brats_path))[1]
    for subject in subjects:
        print(f'sujeto = {subject[:]}')
        p = os.path.join(brats_path, subject, subject+'_t1.nii')
        try:
            control = nib.load(p)
        except:
            print(f'No se encontro MRI para este estudio/paciente')
            continue
        control = nibabel.processing.conform(control, (128, 128, 64))
        control = control.get_fdata()
        control = np.expand_dims(control, 0)
        print(f'{model.predict(control)}')

elif ds == 'NFBS':
    nfbs_path = os.path.join('/media',
                             'davidjm',
                             'Disco_Compartido',
                             'david',
                             'datasets',
                             'NFBS_Dataset'
    )

    subjects = next(os.walk(nfbs_path))[1]
    for subject in subjects:
        print(f'sujeto = {subject[:]}')
        p = os.path.join(nfbs_path, subject, 'sub-'+subject+'_ses-NFB3_T1w_brain.nii.gz')
        try:
            control = nib.load(p)
        except:
            print(f'No se encontro MRI para este estudio/paciente')
            continue
        print(f'control = {control.shape}')
        control = nibabel.processing.conform(control, (128, 128, 64))
        control = np.float32(control.get_fdata())#.astype("float32")
        control = np.expand_dims(control, 0)
        print(f'{model.predict(control)}')



