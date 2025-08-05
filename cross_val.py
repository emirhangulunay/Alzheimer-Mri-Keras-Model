import os
import numpy as np
import matplotlib.pyplot as plt
import datetime

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,
                                     Dropout, BatchNormalization, Activation, InputLayer)
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import KFold
from sklearn.utils import shuffle

data_dir = '/content/drive/MyDrive/Data'
image_size = (248, 496)
batch_size = 64
epochs = 30
patience = 5
learning_rate = 0.0005
weight_decay = 1e-4
k_folds = 5
seed = 42

file_paths = []
labels = []

class_names = sorted(os.listdir(data_dir))
num_classes = len(class_names)
class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}

for cls_name in class_names:
    cls_folder = os.path.join(data_dir, cls_name)
    files = [os.path.join(cls_folder, f) for f in os.listdir(cls_folder) if f.lower().endswith(('png','jpg','jpeg'))]
    file_paths.extend(files)
    labels.extend([class_to_idx[cls_name]] * len(files))

file_paths, labels = shuffle(file_paths, labels, random_state=seed)

print(f"Toplam örnek sayısı: {len(file_paths)}")
print(f"Sınıf isimleri: {class_names}")

def paths_to_dataset(paths, labels, batch_size, shuffle_ds=True):
    def load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=1, dtype=tf.float32)  # grayscale float32
        img = tf.image.resize(img, image_size)
        return img, label

    path_ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    image_label_ds = path_ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle_ds:
        image_label_ds = image_label_ds.shuffle(buffer_size=1000, seed=seed)
    image_label_ds = image_label_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return image_label_ds

def create_model():
    model = Sequential([
        InputLayer(input_shape=(image_size[0], image_size[1], 1)),

        Conv2D(32, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)),
        BatchNormalization(),
        Activation('swish'),
        MaxPooling2D(2, 2),

        Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)),
        BatchNormalization(),
        Activation('swish'),
        MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(weight_decay)),
        BatchNormalization(),
        Activation('swish'),
        MaxPooling2D(2, 2),

        Flatten(),

        Dense(128, kernel_regularizer=l2(weight_decay)),
        BatchNormalization(),
        Activation('swish'),
        Dropout(0.5),

        Dense(64, kernel_regularizer=l2(weight_decay)),
        BatchNormalization(),
        Activation('swish'),
        Dropout(0.5),

        Dense(num_classes, activation='softmax')
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed)

fold_no = 1
acc_per_fold = []
loss_per_fold = []

for train_index, val_index in kf.split(file_paths):
    print(f"\n\n--- Fold {fold_no} ---")

    train_paths = [file_paths[i] for i in train_index]
    train_labels = [labels[i] for i in train_index]
    val_paths = [file_paths[i] for i in val_index]
    val_labels = [labels[i] for i in val_index]

    train_ds = paths_to_dataset(train_paths, train_labels, batch_size, shuffle_ds=True)
    val_ds = paths_to_dataset(val_paths, val_labels, batch_size, shuffle_ds=False)

    model = create_model()

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        verbose=1
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stop, reduce_lr],
        verbose=2
    )

    scores = model.evaluate(val_ds, verbose=0)
    print(f"Fold {fold_no} — Loss: {scores[0]:.4f}, Accuracy: {scores[1]:.4f}")
    acc_per_fold.append(scores[1])
    loss_per_fold.append(scores[0])

    fold_no += 1

print(f"\n{k_folds}-Fold Cross Validation sonuçları:")
print(f"Ortalama doğruluk: {np.mean(acc_per_fold):.4f} ± {np.std(acc_per_fold):.4f}")
print(f"Ortalama kayıp: {np.mean(loss_per_fold):.4f} ± {np.std(loss_per_fold):.4f}")

