# cross_val.py
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation, InputLayer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from google.colab import drive

drive.mount('/content/drive')

# Ayarlar
data_dir = '/content/drive/MyDrive/Data'
image_size = (248, 496)
batch_size = 64
epochs = 20
learning_rate = 0.0005
k = 5  # Fold sayısı

# Dataset klasöründeki tüm dosyaları array'e çevir
dataset = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    image_size=image_size,
    batch_size=1,
    color_mode='grayscale',
    shuffle=False
)

class_names = dataset.class_names
num_classes = len(class_names)

images = []
labels = []

for image_batch, label_batch in dataset:
    images.append(image_batch[0].numpy())
    labels.append(label_batch[0].numpy())

X = np.array(images)
y = np.array(labels)

X = X / 255.0  # Normalizasyon
X = X.reshape(-1, image_size[0], image_size[1], 1)

kf = KFold(n_splits=k, shuffle=True, random_state=42)

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

        Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(0.001)),
        Activation('swish'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(0.001)),
        Activation('swish'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Flatten(),

        Dense(256, activation='swish', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),

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

fold = 1
acc_scores = []

for train_index, val_index in kf.split(X):
    print(f"\n=== Fold {fold} ===")
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    model = create_model()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    _, acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Fold {fold} Accuracy: {acc:.4f}")
    acc_scores.append(acc)
    fold += 1

print(f"\nOrtalama Accuracy: {np.mean(acc_scores):.4f} ± {np.std(acc_scores):.4f}")
