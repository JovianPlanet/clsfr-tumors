import os

def get_parameters(mode):

    # mode = 'reg' # available modes: 'reg', 'train', 'test'

    reg_mode = 'iatm-ctrls' # Options: iatm-ctrls, nfbs

    model_dims = (120, 120, 77) # (128, 128, 64)
    lr         = 0.0001
    epochs     = 300
    batch_size = 4
    new_z      = [2, 2, 2]
    n_heads    = 125
    n_train    = 100
    n_val      = 13
    n_test     = 12

    test_ds = 'IATM-controls' # 'IATM-controls', 'brats', 'reg-NFBS'

    model_fn = 'prueba_.h5'

    brats_train = os.path.join('/home',
                              'davidjm',
                              'Downloads',
                              'BraTS-dataset',
                              'BraTS2020_TrainingData',
                              'MICCAI_BraTS2020_TrainingData'
    )

    brats_val = os.path.join('/home',
                             'davidjm',
                             'Downloads',
                             'BraTS-dataset',
                             'BraTS2020_ValidationData',
                             'MICCAI_BraTS2020_ValidationData'
    )

    nfbs_path = os.path.join('/media',
                             'davidjm',
                             'Disco_Compartido',
                             'david',
                             'datasets',
                             'NFBS_Dataset'
    )

    # Ruta de salida de la imagen registrada
    reg_nfbs_path = os.path.join('/home',
                           'davidjm',
                           'Downloads',
                           'Reg-NFBS'
    )

    # imagen iatm control con resample
    iatm_ctrl_path = os.path.join('/media',
                             'davidjm',
                             'Disco_Compartido',
                             'david',
                             'datasets',
                             'IATM-Dataset',
                             'sujetos_proyecto_controles'
    )

    reg_iatm_ctrl_path = os.path.join('/home',
                           'davidjm',
                           'Downloads',
                           'Reg-IATM',
                           'sujetos_proyecto_controles'
    )

    if mode == 'reg':

        ref_patient  = 'BraTS20_Training_001'
        ref_filename = 'BraTS20_Training_001_t1.nii'

        transforms = ['Translation', # 0
                      'Rigid',       # 1
                      'Similarity',  # 2
                      'QuickRigid',  # 3
                      'DenseRigid',  # 4
                      'BOLDRigid',   # 5
                      'Affine',      # 6
                      'AffineFast',  # 7
                      'BOLDAffine',  # 8
                      'TRSAA',       # 9
                      'ElasticSyN'   # 10
                     ]


        # Ruta de la imagen de referencia
        ref_mri = os.path.join(brats_train,
                               ref_patient, 
                               ref_filename
        )



        return {'mode'               : mode,
                'reg_mode'           : reg_mode,
                'brats_train'        : brats_train,
                'nfbs_path'          : nfbs_path,
                'ref_mri'            : ref_mri, 
                'reg_nfbs_path'      : reg_nfbs_path, 
                'transforms'         : transforms,
                'iatm_ctrl_path'     : iatm_ctrl_path,
                'reg_iatm_ctrl_path' : reg_iatm_ctrl_path
        }

    elif mode == 'train':

        res_path = os.path.join('/media',
                                'davidjm',
                                'Disco_Compartido',
                                'david',
                                'clsfr-tumors',
                                'results'
        )

        return {'mode'          : mode,
                'brats_train'   : brats_train,
                'brats_val'     : brats_val,
                'reg_nfbs_path' : reg_nfbs_path,
                'model_dims'    : model_dims,
                'lr'            : lr,
                'epochs'        : epochs,
                'batch_size'    : batch_size,
                'new_z'         : new_z,
                'n_heads'       : n_heads,
                'n_train'       : n_train,
                'n_val'         : n_val,
                'n_test'        : n_test,
                'model_fn'      : model_fn,
                'res_path'      : res_path, 
        }

    elif mode == 'test':

        threshold = 0.5

        return {'mode'               : mode,
                'brats_train'        : brats_train,
                'brats_val'          : brats_val,
                'reg_nfbs_path'      : reg_nfbs_path,
                'model_dims'         : model_dims,
                'lr'                 : lr,
                'epochs'             : epochs,
                'batch_size'         : batch_size,
                'new_z'              : new_z,
                'n_heads'            : n_heads,
                'n_train'            : n_train,
                'n_val'              : n_val,
                'n_test'             : n_test,
                'thres'              : threshold,
                'model_fn'           : model_fn,
                'test_ds'            : test_ds, 
                'reg_iatm_ctrl_path' : reg_iatm_ctrl_path
        }