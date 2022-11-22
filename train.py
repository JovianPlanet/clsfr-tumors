import os
import numpy as np
import random
from scipy import ndimage

import tensorflow as tf
from tensorflow import keras

from preprocess import preprocess
from cnn import cnn

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

def train(config):

    # Formar lista con las rutas de los pacientes de la base de datos BraTS
    cancer_subjects_path = []
    subjects = next(os.walk(config['brats_train']))[1]
    for subject in subjects:
        p = os.path.join(config['brats_train'], subject, subject+'_t1.nii')
        cancer_subjects_path.append(p)

    print(f'Cancer subjects = {len(cancer_subjects_path)}\n')

    # Formar lista con las rutas de los controles de la base de datos NFBS
    healthy_subjects_path = []
    subjects = next(os.walk(config['reg_nfbs_path']))[1]
    for subject in subjects:
        reg_fn = 'sub-'+subject+'_ses-NFB3_T1w_brain.nii.gz'
        p = os.path.join(config['reg_nfbs_path'], subject, reg_fn)
        healthy_subjects_path.append(p)

    print(f'Cancer subjects = {len(healthy_subjects_path)}\n')

    cancer_scans = np.array([preprocess(path, config) for path in cancer_subjects_path])
    normal_scans = np.array([preprocess(path, config) for path in healthy_subjects_path])

    # Vectores de etiquetas
    cancer_labels = np.array([1 for _ in range(len(cancer_scans))])
    normal_labels = np.array([0 for _ in range(len(normal_scans))])

    # Dividir los datos en train validacion y test
    x_train = np.concatenate((cancer_scans[:100], normal_scans[:100]), axis=0)
    y_train = np.concatenate((cancer_labels[:100], normal_labels[:100]), axis=0)

    x_val = np.concatenate((cancer_scans[100:113], normal_scans[100:113]), axis=0)
    y_val = np.concatenate((cancer_labels[100:113], normal_labels[100:113]), axis=0)

    x_test = np.concatenate((cancer_scans[113:125], normal_scans[113:125]), axis=0)
    y_test = np.concatenate((cancer_scans[113:125], normal_scans[113:125]), axis=0)

    print(f'xtrain shape = {x_train.shape}')

    # Define data loaders.
    train_loader = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    validation_loader = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_loader = tf.data.Dataset.from_tensor_slices((x_test, y_test))

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

    # Build model.
    model = cnn(width=128, height=128, depth=64)
    model.summary()

    # Compile model.
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        config['lr'], decay_steps=100000, decay_rate=0.96, staircase=True
    )
    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
        metrics=["acc"],
    )

    # Define callbacks.
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        "tumor_clf.h5", save_best_only=True
    )
    early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)

    # Train the model, doing validation at the end of each epoch
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=config['epochs'],
        shuffle=True,
        verbose=2,
        callbacks=[checkpoint_cb, early_stopping_cb],
    )


    # fig, ax = plt.subplots(1, 2, figsize=(20, 3))
    # ax = ax.ravel()

    # for i, metric in enumerate(["acc", "loss"]):
    #     ax[i].plot(model.history.history[metric])
    #     ax[i].plot(model.history.history["val_" + metric])
    #     ax[i].set_title("Model {}".format(metric))
    #     ax[i].set_xlabel("epochs")
    #     ax[i].set_ylabel(metric)
    #     ax[i].legend(["train", "val"])

    # Load best weights.
    model.load_weights("tumor_clf.h5")
    prediction = model.predict(np.expand_dims(x_val[1], axis=0))[0]
    scores = [1 - prediction[0], prediction[0]]

    class_names = ["normal", "cancer"]
    for score, name in zip(scores, class_names):
        print(
            "This model is %.2f percent confident that MRI scan is %s"
            % ((100 * score), name)
        )

