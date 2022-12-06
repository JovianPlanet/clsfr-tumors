import os
import numpy as np
from tensorflow import keras
from skimage import transform

import nibabel as nib
import nibabel.processing

from preprocess import preprocess

def test(config):

    model_path = os.path.join('/media',
                              'davidjm',
                              'Disco_Compartido',
                              'david',
                              'clsfr-tumors',
                              config['model_fn']
    )

    model = keras.models.load_model(model_path)
    print(f'Model summary = {model.summary()}')
    print(f'Model layers = {model.layers}')

    print(f'\nPredicciones del modelo {config["model_fn"]} para el dataset {config["test_ds"]}:\n')

    predictions = []
    if config['test_ds'] == 'IATM-controls':

        subjects = next(os.walk(config['reg_iatm_ctrl_path']))[1]
        for subject in subjects:
            studies = next(os.walk(os.path.join(config['reg_iatm_ctrl_path'], subject)))[1]
            p = os.path.join(config['reg_iatm_ctrl_path'], subject, studies[0], 'NIFTI', 'sub-'+subject[:-4]+'_T1w.nii.gz')
            print(f'p = {p}')
            try:
                control = preprocess(p, config)
            except:
                print(f'No se encontro MRI para este estudio/paciente')
                continue
            control = np.expand_dims(control, 0)
            print(f'pred = {model.predict(control)[0][0]:.3f}')
            if model.predict(control)[0][0] <= config['thres']:
                predictions.append(1)
            else:
                predictions.append(0)
        print(f'\nAccuracy = {sum(predictions)/len(predictions)}')

    elif config['test_ds'] == 'brats':

        subjects = next(os.walk(config['brats_val']))[1]
        for subject in subjects:
            p = os.path.join(config['brats_val'], subject, subject+'_t1.nii')
            try:
                control = preprocess(p, config)
            except:
                print(f'No se encontro MRI para este estudio/paciente')
                continue
            control = np.expand_dims(control, 0)
            if model.predict(control)[0][0] <= config['thres']:
                predictions.append(0)
            else:
                predictions.append(1)
        print(f'\nAccuracy = {sum(predictions)/len(predictions)}')

    elif config['test_ds'] == 'reg-NFBS':

        subjects = next(os.walk(config['reg_nfbs_path']))[1]
        for subject in subjects:
            p = os.path.join(config['reg_nfbs_path'], subject, 'sub-'+subject+'_ses-NFB3_T1w_brain.nii.gz')
            try:
                control = preprocess(p, config)
            except:
                print(f'No se encontro MRI para este estudio/paciente')
                continue
            control = np.expand_dims(control, 0)
            if model.predict(control)[0][0] <= config['thres']:
                predictions.append(1)
            else:
                predictions.append(0)
        print(f'\nAccuracy = {sum(predictions)/len(predictions)}')



