import os
import numpy as np
import matplotlib.pyplot as plt
import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,
                                     Dropout, BatchNormalization, Activation, InputLayer)
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from google.colab import drive

drive.mount('/content/drive')

print("TF version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

data_dir = '/content/drive/MyDrive/Data'
image_size = (128, 128)
batch_size = 64
epochs = 20
patience = 3
learning_rate = 0.0005
val_split = 0.2
seed = 42

log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

raw_train_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=val_split,
    subset='training',
    seed=seed,
    image_size=image_size,
    batch_size=batch_size,
    color_mode='grayscale'
)

raw_val_dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=val_split,
    subset='validation',
    seed=seed,
    image_size=image_size,
    batch_size=batch_size,
    color_mode='grayscale'
)

class_names = raw_train_dataset.class_names
num_classes = len(class_names)
print("Sınıf isimleri:", class_names)

normalization_layer = tf.keras.layers.Rescaling(1./255)
train_dataset = raw_train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = raw_val_dataset.map(lambda x, y: (normalization_layer(x), y))

def create_model():
    model = Sequential([
        InputLayer(input_shape=(image_size[0], image_size[1], 1)),

        Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(0.001)),
        Activation('swish'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(0.001)),
        Activation('swish'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Flatten(),

        Dense(256, activation='swish', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),

        Dense(64, activation='swish', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),

        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

model = create_model()
model.summary()

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=patience,
    restore_best_weights=True
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=[early_stop, tensorboard_callback]
)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

model.save("/content/drive/MyDrive/alzheimer_mri_model.h5")