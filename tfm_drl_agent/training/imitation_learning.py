import pandas as pd
import tensorflow as tf
import os
import glob

def load_expert_data_multiple(folder_path):
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    print(f"[INFO] Cargando {len(all_files)} archivos de datos de experto desde: {folder_path}")

    X_list, y_list = [], []

    for file in all_files:
        df = pd.read_csv(file)
        if df.empty:
            print(f"[WARNING] Archivo vacío ignorado: {file}")
            continue

        try:
            X = df[["latitude", "longitude", "heading", "speed", "wind_dir", "wind_speed"]].values
            y = df[["target_heading", "target_speed"]].values

            # Escalado etiquetas a [-1,1]
            y[:, 0] = (y[:, 0] / 180.0) - 1.0
            y[:, 1] = (y[:, 1] / 10.0) - 1.0

            X_list.append(X)
            y_list.append(y)
        except KeyError as e:
            print(f"[ERROR] Faltan columnas esperadas en {file}: {e}")
            continue

    if not X_list:
        raise ValueError("No se pudieron cargar datos válidos de ningún archivo.")

    X_all = pd.concat([pd.DataFrame(x) for x in X_list], ignore_index=True).values
    y_all = pd.concat([pd.DataFrame(y) for y in y_list], ignore_index=True).values

    print(f"[INFO] Total de muestras cargadas: {X_all.shape[0]}")
    return X_all, y_all


def train_actor_supervised(actor_model, folder_path, epochs=50, batch_size=64, learning_rate=1e-4):
    X, y = load_expert_data_multiple(folder_path)

    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.shuffle(1000).batch(batch_size).repeat()

    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss_fn = tf.keras.losses.MeanSquaredError()

    steps_per_epoch = X.shape[0] // batch_size

    for epoch in range(epochs):
        epoch_loss = 0
        for step, (xb, yb) in enumerate(dataset.take(steps_per_epoch)):
            with tf.GradientTape() as tape:
                pred = actor_model(xb, training=True)
                loss = loss_fn(yb, pred)
            grads = tape.gradient(loss, actor_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, actor_model.trainable_variables))
            epoch_loss += loss.numpy()

        avg_loss = epoch_loss / steps_per_epoch
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    return actor_model
