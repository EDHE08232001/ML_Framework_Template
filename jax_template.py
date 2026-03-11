# ===============================
# Imports
# ===============================
import os
import yaml
import time
import gc
from typing import Dict, Tuple, Any

import jax
import jax.numpy as jnp
from jax import random
import flax.linen as nn
from flax.training import train_state
import optax

from tensorflow.keras.datasets import mnist
import numpy as np


# =====================================================
# 1) Reproducibility Utilities
# =====================================================
def seed_everything(seed: int):
    """
    Set seeds for reproducibility.
    """
    np.random.seed(seed)
    return random.key(seed)


def get_device():
    """
    JAX automatically uses available accelerators (GPU/TPU/MPS).
    """
    devices = jax.devices()
    return devices[0].platform


# =====================================================
# 1.5) Memory Cleanup
# =====================================================
def cleanup():
    """Force garbage collection."""
    gc.collect()


# =====================================================
# 2) Neural Network Model (using Flax)
# =====================================================
class MLP(nn.Module):
    """
    Multi-Layer Perceptron using Flax.
    """
    input_dim: int = 784
    hidden_dim: int = 256
    output_dim: int = 10
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, training: bool = False):
        x = x.reshape((x.shape[0], -1))  # Flatten
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        
        if self.dropout > 0.0:
            # Dropout in Flax requires an explicit PRNG collection during apply
            x = nn.Dropout(rate=self.dropout, deterministic=not training)(x)
        
        x = nn.Dense(self.output_dim)(x)
        return x


# =====================================================
# 3) Data Pipeline
# =====================================================
def make_loader(batch_size: int, key: jax.random.PRNGKey):
    """Load and batch MNIST data."""
    (x_train, y_train), _ = mnist.load_data()
    
    x_train = x_train.astype(np.float32) / 255.0
    y_train = y_train.astype(np.int32)
    
    n = len(x_train)
    indices = np.arange(n)
    key, subkey = random.split(key)
    indices = random.permutation(subkey, indices)
    
    x_train = x_train[indices]
    y_train = y_train[indices]
    
    num_batches = n // batch_size
    x_batches = x_train[:num_batches * batch_size].reshape(
        num_batches, batch_size, 28, 28
    )
    y_batches = y_train[:num_batches * batch_size].reshape(
        num_batches, batch_size
    )
    
    return x_batches, y_batches, key


# =====================================================
# 4) Training Step (Functional & JIT Compiled)
# =====================================================
@jax.jit
def train_step(state: train_state.TrainState, x, y, dropout_key):
    """
    Single training step. @jax.jit compiles this entire function 
    into a highly optimized static XLA graph.
    """
    def loss_fn(params):
        # We use state.apply_fn instead of passing the model object directly
        logits = state.apply_fn(
            {"params": params}, 
            x, 
            training=True,
            rngs={"dropout": dropout_key} # Pass the dropout key here
        )
        y_onehot = jax.nn.one_hot(y, num_classes=10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits, y_onehot))
        return loss, logits

    # Compute loss and gradients simultaneously 
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss_value, logits), grads = grad_fn(state.params)
    
    # Immutable update: returns a NEW state object
    new_state = state.apply_gradients(grads=grads)
    
    # Calculate accuracy
    pred = jnp.argmax(logits, axis=1)
    acc = jnp.mean(pred == y)
    
    return new_state, loss_value, acc


# =====================================================
# 5) One Full Experiment Run
# =====================================================
def run_experiment(cfg: Dict, run_id: int, seed: int):
    key = seed_everything(seed)
    
    model = MLP(
        input_dim=cfg["model"]["input_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        output_dim=cfg["model"]["output_dim"],
        dropout=cfg["model"]["dropout"],
    )
    
    # We need multiple keys for parameter init vs. dropout init
    key, init_key, dropout_init_key = random.split(key, 3)
    
    # FIX: Initialize dummy input with the actual batch size
    batch_size = cfg["dataset"]["batch_size"]
    dummy_input = jnp.ones((batch_size, 28, 28), dtype=jnp.float32)
    
    # Initialize the model weights
    variables = model.init(
        {"params": init_key, "dropout": dropout_init_key}, 
        dummy_input, 
        training=False
    )
    params = variables["params"]
    
    tx = optax.adam(learning_rate=cfg["training"]["learning_rate"])
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )
    
    x_batches, y_batches, key = make_loader(batch_size, key)
    
    num_batches = len(x_batches)
    total_steps = cfg["training"]["epochs"] * num_batches
    mid_step = total_steps // 2
    step = 0
    
    log = []
    
    os.makedirs(cfg["paths"]["checkpoints_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["logs_dir"], exist_ok=True)
    
    for epoch in range(cfg["training"]["epochs"]):
        for batch_idx in range(num_batches):
            step += 1
            
            x = x_batches[batch_idx]
            y = y_batches[batch_idx]
            
            # Split a new key at every single step to ensure unique dropout masks
            key, step_dropout_key = random.split(key)
            
            state, loss_value, acc = train_step(state, x, y, step_dropout_key)
            
            log.append({
                "step": step,
                "loss": float(loss_value),
                "acc": float(acc)
            })
            
            if step == mid_step:
                with open(f"{cfg['paths']['checkpoints_dir']}/run{run_id}_mid.npy", "wb") as f:
                    np.save(f, state.params)
    
    with open(f"{cfg['paths']['checkpoints_dir']}/run{run_id}_final.npy", "wb") as f:
        np.save(f, state.params)
    
    cleanup()
    return log


# =====================================================
# 6) Entry Point
# =====================================================
def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    
    print(f"Using device: {get_device()}")
    print(f"Running {cfg['num_runs']} experiments...\n")
    
    all_runs = []
    
    for i in range(cfg["num_runs"]):
        run_id = i + 1
        seed = cfg["seed_base"] + i
        print(f"[Run {run_id}] seed={seed}")
        
        log = run_experiment(cfg, run_id, seed)
        all_runs.append(log)
        
        cleanup()
        
        out = os.path.join(cfg["paths"]["logs_dir"], f"run{run_id}.log")
        with open(out, "w") as f:
            f.write("step,loss,acc\n")
            for row in log:
                f.write(f"{row['step']},{row['loss']},{row['acc']}\n")
        
        print(f"  Saved to {out}\n")
    
    steps = len(all_runs[0])
    arr = np.zeros((len(all_runs), steps, 2), dtype=np.float32)
    
    for r in range(len(all_runs)):
        for t in range(steps):
            arr[r, t, 0] = all_runs[r][t]["loss"]
            arr[r, t, 1] = all_runs[r][t]["acc"]
    
    np.save(os.path.join(cfg["paths"]["logs_dir"], "all_runs.npy"), arr)
    print("All runs complete. Metrics saved to logs/all_runs.npy")


if __name__ == "__main__":
    main()
