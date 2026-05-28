import os
import json
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from model import build_pinn

def train_pinn(project_dir, use_large=False):
    # 1. Load datasets and normalization stats
    data_dir = os.path.join(project_dir, 'data')
    
    train_file = 'train_data_large.json' if use_large else 'train_data.json'
    val_file = 'val_data_large.json' if use_large else 'val_data.json'
    stats_file = 'normalization_stats_large.json' if use_large else 'normalization_stats.json'
    log_file = 'training_log_large.csv' if use_large else 'training_log_multi_re.csv'
    model_file = 'pinn_model_large.keras' if use_large else 'pinn_model.keras'
    
    with open(os.path.join(data_dir, train_file), 'r') as f:
        train_json = json.load(f)
    with open(os.path.join(data_dir, val_file), 'r') as f:
        val_json = json.load(f)
    with open(os.path.join(data_dir, stats_file), 'r') as f:
        stats = json.load(f)
        
    X_train = np.array(train_json['X_train'], dtype=np.float32)
    Y_train = np.array(train_json['Y_train'], dtype=np.float32)
    X_val = np.array(val_json['X_val'], dtype=np.float32)
    Y_val = np.array(val_json['Y_val'], dtype=np.float32)
    
    # Normalization statistics as tensors
    X_mean = tf.constant(stats['X_mean'], dtype=tf.float32)
    X_std = tf.constant(stats['X_std'], dtype=tf.float32)
    Y_mean = tf.constant(stats['Y_mean'], dtype=tf.float32)
    Y_std = tf.constant(stats['Y_std'], dtype=tf.float32)
    
    # 2. Hyperparameters
    LAMBDA_PHYSICS = 0.05
    BATCH_SIZE = 64
    EPOCHS = 500 if use_large else 5000 # Scaled down for large dataset to maintain similar training steps
    LR_MAX = 1e-3
    LR_MIN = 1e-5
    PRANDTL_LIMIT = 4.0 * math.pi
    
    # 3. Model & Optimizer
    model = build_pinn()
    optimizer = keras.optimizers.Adam(learning_rate=LR_MAX)
    
    # TensorFlow Constants for the graph
    prandtl_tf = tf.constant(PRANDTL_LIMIT, dtype=tf.float32)
    
    @tf.function
    def train_step(x_batch, y_batch):
        with tf.GradientTape() as tape:
            y_pred = model(x_batch, training=True)
            
            # Data MSE Loss
            data_loss = tf.reduce_mean(tf.square(y_pred - y_batch))
            
            # Un-normalize predictions to physical space
            y_phys = y_pred * Y_std + Y_mean
            cd_pred = y_phys[:, 0]
            cl_pred = y_phys[:, 1]
            
            # Un-normalize alpha for physics checks
            alpha_raw = x_batch[:, 0] * X_std[0] + X_mean[0]
            
            # 1. Magnus Effect violation: cl_pred * alpha_raw < 0
            magnus_violation = tf.nn.relu(-cl_pred * alpha_raw)
            magnus_loss = tf.reduce_mean(tf.square(magnus_violation))
            
            # 2. Zero-lift at rest: alpha = 0 -> Cl = 0
            zero_lift_mask = tf.cast(tf.abs(alpha_raw) < 0.1, tf.float32)
            zero_lift_loss = tf.reduce_mean(tf.square(cl_pred * zero_lift_mask))
            
            # 3. Drag positivity: Cd >= 0
            drag_violation = tf.nn.relu(-cd_pred)
            drag_loss = tf.reduce_mean(tf.square(drag_violation))
            
            # 4. Prandtl upper bound: |Cl| <= 4*pi
            prandtl_violation = tf.nn.relu(tf.abs(cl_pred) - prandtl_tf)
            prandtl_loss = tf.reduce_mean(tf.square(prandtl_violation))
            
            # Total physics loss
            physics_loss = magnus_loss + zero_lift_loss + drag_loss + prandtl_loss
            
            # Total combined loss
            total_loss = data_loss + LAMBDA_PHYSICS * physics_loss
            
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        return total_loss, data_loss, physics_loss
    
    # 4. Training Loop
    print(f"Starting training for {EPOCHS} epochs on {'LARGE' if use_large else 'DEFAULT'} dataset...")
    print(f"Physics loss weight (lambda): {LAMBDA_PHYSICS}")
    print(f"Learning rate: Cosine decay {LR_MAX} -> {LR_MIN}")
    print(f"Batch size: {BATCH_SIZE}")
    
    history = {'epoch': [], 'loss': [], 'data_loss': [], 'phys_loss': [], 'val_mse': []}
    
    dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    
    for epoch in range(1, EPOCHS + 1):
        # Calculate learning rate from cosine decay schedule
        lr = LR_MIN + (LR_MAX - LR_MIN) * 0.5 * (1.0 + math.cos(math.pi * (epoch - 1) / EPOCHS))
        optimizer.learning_rate.assign(lr)
        
        # Shuffle and batch
        shuffled_dataset = dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)
        
        epoch_total = 0.0
        epoch_data = 0.0
        epoch_phys = 0.0
        batches = 0
        
        for x_batch, y_batch in shuffled_dataset:
            total_loss, data_loss, phys_loss = train_step(x_batch, y_batch)
            epoch_total += total_loss.numpy()
            epoch_data += data_loss.numpy()
            epoch_phys += phys_loss.numpy()
            batches += 1
            
        # Validation evaluation
        y_val_pred = model(X_val, training=False)
        val_mse = tf.reduce_mean(tf.square(y_val_pred - Y_val)).numpy()
        
        # Average losses
        epoch_total /= batches
        epoch_data /= batches
        epoch_phys /= batches
        
        # Save metrics
        history['epoch'].append(epoch)
        history['loss'].append(epoch_total)
        history['data_loss'].append(epoch_data)
        history['phys_loss'].append(epoch_phys)
        history['val_mse'].append(val_mse)
        
        if epoch % (20 if use_large else 200) == 0 or epoch == 1 or epoch == EPOCHS:
            print(f"Epoch {epoch:4d}/{EPOCHS} - loss: {epoch_total:.5f} | data: {epoch_data:.5f} | phys: {epoch_phys:.6f} | val_mse: {val_mse:.5f} | lr: {lr:.2e}")
            
    # 5. Save Model and History
    models_dir = os.path.join(project_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, model_file)
    model.save(model_path)
    print(f"Model saved to {model_path}")
    
    history_df = pd.DataFrame(history)
    history_path = os.path.join(data_dir, log_file)
    history_df.to_csv(history_path, index=False)
    print(f"Training history saved to {history_path}")
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train PINN")
    parser.add_argument('--large', action='store_true', help="Train using the large dataset (~100k rows)")
    args = parser.parse_args()
    
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_pinn(project_dir, use_large=args.large)
