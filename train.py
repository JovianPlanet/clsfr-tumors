import os
from tensorflow import keras

from get_data import get_data
from cnn import cnn

def train(config):

    train_dataset, validation_dataset, test_dataset = get_data(config)

    # Build model.7875
    model = cnn(120, 120, 77)
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
    checkpoint_cb     = keras.callbacks.ModelCheckpoint(config['model_fn'], save_best_only=True)
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

    print(f'\nEvaluacion del modelo:\n {model.evaluate(test_dataset)}')



