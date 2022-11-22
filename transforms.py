import os
from pathlib import Path
import nibabel as nib
from ants import image_read, image_write, registration, apply_transforms
from nipype.interfaces.fsl import BET

def registrate_NFBS(config):

    # Cargar las imagenes
    ref = image_read(config['ref_mri'], pixeltype='unsigned int')

    subjects = next(os.walk(config['nfbs_path']))[1]

    for subject in subjects[:]:

        print(f'NFBS Subject = {subject}')

        reg_fn = 'sub-'+subject+'_ses-NFB3_T1w_brain.nii.gz'
        p = os.path.join(config['nfbs_path'], subject, reg_fn)

        try:
            reg_img = image_read(p, pixeltype='unsigned int')
        except:
            print(f'No se encontro MRI para este estudio/paciente')
            continue

        # Registrar imagen
        rs2_img = registration(fixed=ref, 
                               moving=reg_img, 
                               type_of_transform=config['transforms'][4]
        )

        rs = apply_transforms(fixed=ref, 
                              moving=reg_img, 
                              transformlist=rs2_img['fwdtransforms'], 
                              interpolator='linear' #'multiLabel'
        )

        Path(os.path.join(config['out_reg'], subject)).mkdir(parents=True, exist_ok=True)
        image_write(rs, os.path.join(config['out_reg'], subject, reg_fn), ri=False)

def reg_IATM_controls(config):

    # Cargar MRI de referencia
    ref = image_read(config['ref_mri'], pixeltype='unsigned int')

    subjects = next(os.walk(config['iatm_ctrl_path']))[1]
    for subject in subjects[:]:

        print(f'sujeto = {subject[:-4]}')
        studies = next(os.walk(os.path.join(config['iatm_ctrl_path'], subject)))[1]
        print(f'study = {studies}')
        reg_fn = 'sub-'+subject[:-4]+'_T1w.nii.gz'
        p = os.path.join(config['iatm_ctrl_path'], subject, studies[0], 'NIFTI', reg_fn)

        out_dir = os.path.join(config['reg_iatm_ctrl_path'], subject, studies[0], 'NIFTI')
        brain = 'brain-'+reg_fn
        b = os.path.join(out_dir, brain)

        Path(out_dir).mkdir(parents=True, exist_ok=True)

        try:
            fsl_bet(p, b)
        except:
            print(f'No se encontro MRI para este estudio/paciente')
            continue
        try:
            reg_img = image_read(b, pixeltype='unsigned int')
        except:
            print(f'No se encontro MRI para este estudio/paciente')
            continue

        # Registrar imagen
        rs2_img = registration(fixed=ref, 
                               moving=reg_img, 
                               type_of_transform=config['transforms'][4]
        )

        rs = apply_transforms(fixed=ref, 
                              moving=reg_img, 
                              transformlist=rs2_img['fwdtransforms'], 
                              interpolator='linear' #'multiLabel'
        )

        os.remove(b)
        image_write(rs, os.path.join(out_dir, reg_fn), ri=False)

'''
Performs FSL's skull striping
'''
def fsl_bet(input_file, out_file):
    skullstrip = BET()
    skullstrip.inputs.in_file = input_file      # os.path.join(head_path, head)
    skullstrip.inputs.out_file = out_file       # os.path.join(brain_path, fsl_brain)
    skullstrip.inputs.frac = 0.4                # Rango = [0,1] Valores mas pequenos estiman un area mayor de cerebro
    skullstrip.inputs.robust = True
    skullstrip.run()