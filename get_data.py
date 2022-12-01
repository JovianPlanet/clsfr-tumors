import os
import numpy as np
import tensorflow as tf
from preprocess import preprocess

@tf.function
def rotate(volume):
    """Rotate the volume by a few degrees"""

    def scipy_rotate(volume):
        # define some rotation angles
        angles = [-20, -10, -5, 5, 10, 20]
        # pick angles at random
        angle = random.choice(angles)
        # rotate volume
        volume = ndimage.rotate(volume, angle, reshape=False)
        volume[volume < 0] = 0
        volume[volume > 1] = 1
        return volume

    augmented_volume = tf.numpy_function(scipy_rotate, [volume], tf.float32)
    return augmented_volume


def train_preprocessing(volume, label):
    """Process training data by rotating and adding a channel."""
    #volume = rotate(volume)
    volume = tf.expand_dims(volume, axis=3)
    return volume, label


def validation_preprocessing(volume, label):
    """Process validation data by only adding a channel."""
    volume = tf.expand_dims(volume, axis=3)
    return volume, label

def get_data(config):

    # Formar lista con las rutas de los pacientes de la base de datos BraTS
    cancer_subjects_path = []
    subjects = next(os.walk(config['brats_train']))[1]
    for subject in subjects:
        p = os.path.join(config['brats_train'], subject, subject+'_t1.nii')
        cancer_subjects_path.append(p)

    print(f'Total cancer subjects = {len(cancer_subjects_path)}\n')

    # Formar lista con las rutas de los controles de la base de datos NFBS
    healthy_subjects_path = []
    subjects = next(os.walk(config['reg_nfbs_path']))[1]
    for subject in subjects:
        reg_fn = 'sub-'+subject+'_ses-NFB3_T1w_brain.nii.gz'
        p = os.path.join(config['reg_nfbs_path'], subject, reg_fn)
        healthy_subjects_path.append(p)

    print(f'Total healthy subjects = {len(healthy_subjects_path)}\n')

    cancer_scans = np.array([preprocess(path, config) for path in cancer_subjects_path[:config['n_heads']]])
    normal_scans = np.array([preprocess(path, config) for path in healthy_subjects_path][:config['n_heads']])

    print(f'Cancer scans shape = {cancer_scans.shape}')
    print(f'Normal scans shape = {normal_scans.shape}')

    # Vectores de etiquetas
    cancer_labels = np.array([1 for _ in range(len(cancer_scans))])
    normal_labels = np.array([0 for _ in range(len(normal_scans))])

    # Dividir los datos en train validacion y test
    x_train = np.concatenate(( cancer_scans[:config['n_train']], 
                               normal_scans[:config['n_train']]), axis=0)
    y_train = np.concatenate((cancer_labels[:config['n_train']], 
                              normal_labels[:config['n_train']]), axis=0)

    x_val = np.concatenate(( cancer_scans[config['n_train']:config['n_train']+config['n_val']], 
                             normal_scans[config['n_train']:config['n_train']+config['n_val']]), axis=0)
    y_val = np.concatenate((cancer_labels[config['n_train']:config['n_train']+config['n_val']], 
                            normal_labels[config['n_train']:config['n_train']+config['n_val']]), axis=0)

    x_test = np.concatenate(( cancer_scans[config['n_train']+config['n_val']:], 
                              normal_scans[config['n_train']+config['n_val']:]), axis=0)
    y_test = np.concatenate((cancer_labels[config['n_train']+config['n_val']:], 
                             normal_labels[config['n_train']+config['n_val']:]), axis=0)

    print(f'\nn train = {x_train.shape[0]}, val = {x_val.shape[0]}, test = {x_test.shape[0]}\n')

    # Define data loaders.
    train_loader      = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_loader       = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    # Augment the on the fly during training.
    train_dataset = (
        train_loader.shuffle(len(x_train))
        .map(train_preprocessing)
        .batch(config['batch_size'])
        .prefetch(2)
    )
    # Only rescale.
    validation_dataset = (
        validation_loader.shuffle(len(x_val))
        .map(validation_preprocessing)
        .batch(config['batch_size'])
        .prefetch(2)
    )

    test_dataset = (
        test_loader.shuffle(len(x_test))
        .map(validation_preprocessing)
        .batch(config['batch_size'])
        .prefetch(2)
    )

    return train_dataset, validation_dataset, test_dataset