from pathlib import Path

import matplotlib.pyplot as plt
from tensorflow.keras import callbacks, optimizers


def setup_callbacks(output_dir="outputs"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    best_model_path = output_path / "deepfake_model_best.h5"
    return [
        callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=False,
            mode="max",
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
    ]


def train_model(model, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, output_dir="outputs"):
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=setup_callbacks(output_dir),
        verbose=1,
    )
    return history


def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history.history["accuracy"], label="Training Accuracy", color="blue")
    ax1.plot(history.history["val_accuracy"], label="Validation Accuracy", color="red")
    ax1.set_title("Model Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history["loss"], label="Training Loss", color="blue")
    ax2.plot(history.history["val_loss"], label="Validation Loss", color="red")
    ax2.set_title("Model Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

