import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')

lr = 0.001
batch = 64
epochs = 10
patience = 5
k = 5

data_dir = '/content/drive/MyDrive/Data'
img_height, img_width = 248, 496  

file_paths = []
labels = []
class_names = sorted(os.listdir(data_dir))
class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}

for cls in class_names:
    cls_folder = os.path.join(data_dir, cls)
    for fname in os.listdir(cls_folder):
        if fname.endswith('.jpg'):
            file_paths.append(os.path.join(cls_folder, fname))
            labels.append(class_to_idx[cls])

file_paths = np.array(file_paths)
labels = np.array(labels)

skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

fold_accuracies, fold_precisions, fold_recalls, fold_f1_scores = [], [], [], []

def build_model(num_classes):
    model = models.Sequential([
        layers.Input(shape=(img_height, img_width, 3)),
        layers.Conv2D(32, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

for fold, (train_idx, val_idx) in enumerate(skf.split(file_paths, labels)):
    print(f"Processing fold {fold+1} of {k}")

    train_files, val_files = file_paths[train_idx], file_paths[val_idx]
    train_labels, val_labels = labels[train_idx], labels[val_idx]

    train_gen = ImageDataGenerator(rescale=1./255)
    val_gen = ImageDataGenerator(rescale=1./255)

    train_df = pd.DataFrame({'filename': train_files, 'class': train_labels})
    val_df = pd.DataFrame({'filename': val_files, 'class': val_labels})

    train_data = train_gen.flow_from_dataframe(
        dataframe=train_df,
        x_col='filename', y_col='class',
        target_size=(img_height, img_width),
        class_mode='raw', batch_size=batch, shuffle=True
    )
    val_data = val_gen.flow_from_dataframe(
        dataframe=val_df,
        x_col='filename', y_col='class',
        target_size=(img_height, img_width),
        class_mode='raw', batch_size=batch, shuffle=False
    )

    model = build_model(num_classes=len(class_names))

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience, restore_best_weights=True, verbose=1)

    model.fit(train_data, epochs=epochs, validation_data=val_data, callbacks=[early_stop], verbose=1)

    val_preds = model.predict(val_data)
    val_preds_classes = np.argmax(val_preds, axis=1)
    val_true = val_labels

    acc = accuracy_score(val_true, val_preds_classes)
    prec = precision_score(val_true, val_preds_classes, average='weighted')
    rec = recall_score(val_true, val_preds_classes, average='weighted')
    f1 = f1_score(val_true, val_preds_classes, average='weighted')

    fold_accuracies.append(acc)
    fold_precisions.append(prec)
    fold_recalls.append(rec)
    fold_f1_scores.append(f1)

    print(f"Fold {fold+1} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

print("\nK-Fold Cross-Validation Results:")
print(f"Average Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
print(f"Average Precision: {np.mean(fold_precisions):.4f} ± {np.std(fold_precisions):.4f}")
print(f"Average Recall: {np.mean(fold_recalls):.4f} ± {np.std(fold_recalls):.4f}")
print(f"Average F1 Score: {np.mean(fold_f1_scores):.4f} ± {np.std(fold_f1_scores):.4f}")

folds = np.arange(1, k+1)

plt.figure(figsize=(10, 6))
plt.plot(folds, fold_accuracies, marker='o', label='Accuracy')
plt.plot(folds, fold_precisions, marker='o', label='Precision')
plt.plot(folds, fold_recalls, marker='o', label='Recall')
plt.plot(folds, fold_f1_scores, marker='o', label='F1 Score')
plt.title('K-Fold Cross-Validation Metrics')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.xticks(folds)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
means = [np.mean(fold_accuracies), np.mean(fold_precisions), np.mean(fold_recalls), np.mean(fold_f1_scores)]
stds = [np.std(fold_accuracies), np.std(fold_precisions), np.std(fold_recalls), np.std(fold_f1_scores)]

plt.figure(figsize=(8, 5))
plt.bar(metrics, means, yerr=stds, capsize=8, color='skyblue')
plt.title('K-Fold Cross-Validation Mean Metrics')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.grid(axis='y')
