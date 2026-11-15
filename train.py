import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42

DATA_DIR = Path("data/chest_xray")
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
MODEL_PATH = Path("models/pneumonia_cnn.keras")
OUTPUT_DIR = Path("outputs")

for required_dir in [TRAIN_DIR / "NORMAL", TRAIN_DIR / "PNEUMONIA",
                     TEST_DIR / "NORMAL", TEST_DIR / "PNEUMONIA"]:
    if not required_dir.exists():
        raise FileNotFoundError(
            f"Missing folder: {required_dir}. "
            "Download and extract the Kaggle dataset as described in README.md."
        )

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="binary",
)

class_names = train_ds.class_names
if class_names != ["NORMAL", "PNEUMONIA"]:
    raise ValueError(f"Unexpected class order: {class_names}")

val_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="binary",
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    label_mode="binary",
    shuffle=False,
)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# Calculate balanced class weights using training-folder image counts.
class_counts = []
for class_name in class_names:
    folder = TRAIN_DIR / class_name
    count = sum(
        1 for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    class_counts.append(count)

if min(class_counts) == 0:
    raise ValueError("One of the training classes contains no images.")

labels_for_weights = np.concatenate([
    np.full(count, class_index, dtype=int)
    for class_index, count in enumerate(class_counts)
])
weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=labels_for_weights,
)
class_weights = {i: float(weight) for i, weight in enumerate(weights)}
print("Class counts:", dict(zip(class_names, class_counts)))
print("Class weights:", class_weights)

# CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 1)),
    tf.keras.layers.Rescaling(1.0 / 255),
    tf.keras.layers.RandomRotation(0.03),
    tf.keras.layers.RandomZoom(0.05),

    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ],
)

model.summary()
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks,
)

# Save training plots.
for metric, val_metric, title, filename in [
    ("accuracy", "val_accuracy", "CNN Model Accuracy", "training_accuracy.png"),
    ("loss", "val_loss", "CNN Model Loss", "training_loss.png"),
]:
    plt.figure(figsize=(8, 5))
    plt.plot(history.history[metric], label=f"Training {metric}")
    plt.plot(history.history[val_metric], label=f"Validation {metric}")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=150)
    plt.close()

# Evaluate once on the untouched test set.
best_model = tf.keras.models.load_model(MODEL_PATH)
y_true = np.concatenate([
    labels.numpy().reshape(-1) for _, labels in test_ds
]).astype(int)
y_prob = best_model.predict(test_ds, verbose=1).ravel()
y_pred = (y_prob >= 0.5).astype(int)

report = classification_report(
    y_true, y_pred, labels=[0, 1],
    target_names=class_names, zero_division=0
)
print("\nClassification report:\n", report)
(OUTPUT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
disp.plot(cmap="Blues", values_format="d")
plt.title("Pneumonia Detection - Confusion Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
plt.close()

try:
    auc = roc_auc_score(y_true, y_prob)
    print(f"ROC-AUC: {auc:.4f}")
    (OUTPUT_DIR / "roc_auc.txt").write_text(f"ROC-AUC: {auc:.4f}\n")
except ValueError as exc:
    print("ROC-AUC could not be calculated:", exc)

print(f"\nTraining and evaluation complete. Model saved to: {MODEL_PATH}")
