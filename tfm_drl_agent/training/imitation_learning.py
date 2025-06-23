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
        X = df[["latitude", "longitude", "heading", "speed", "wind_dir", "wind_speed"]].values
        y = df[["target_heading", "target_speed"]].values

        # Escalar etiquetas a rango [-1, 1] para que coincidan con activación tanh del actor
        y[:, 0] = (y[:, 0] / 180.0) - 1.0     # heading: [0,360] → [-1,1]
        y[:, 1] = (y[:, 1] / 10.0) - 1.0      # speed:   [0,20]  → [-1,1]

        X_list.append(X)
        y_list.append(y)

    X_all = pd.concat([pd.DataFrame(x) for x in X_list], ignore_index=True).values
    y_all = pd.concat([pd.DataFrame(y) for y in y_list], ignore_index=True).values
    return X_all, y_all

def train_actor_supervised(actor_model, folder_path, epochs=50, batch_size=64, learning_rate=1e-4):
    X, y = load_expert_data_multiple(folder_path)
    dataset = tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size).shuffle(1000)

    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss_fn = tf.keras.losses.MeanSquaredError()

    for epoch in range(epochs):
        epoch_loss = 0
        for step, (xb, yb) in enumerate(dataset):
            with tf.GradientTape() as tape:
                pred = actor_model(xb, training=True)
                loss = loss_fn(yb, pred)
            grads = tape.gradient(loss, actor_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, actor_model.trainable_variables))
            epoch_loss += loss.numpy()

        avg_loss = epoch_loss / (step + 1)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    return actor_model
