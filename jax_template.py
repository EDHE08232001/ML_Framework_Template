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
from jax import random, grad, vmap
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
    JAX uses key-based random generation.
    """
    np.random.seed(seed)
    # JAX's PRNG is key-based; create a master key
    return random.key(seed)


def get_device():
    """
    Check available devices and return device type string.
    JAX automatically uses available accelerators (GPU/TPU).
    """
    devices = jax.devices()
    device_type = devices[0].platform
    return device_type


# =====================================================
# 1.5) Memory Cleanup
# =====================================================
def cleanup():
    """
    Force garbage collection.
    JAX is more efficient with memory than PyTorch,
    but we can still explicitly clean up.
    """
    gc.collect()


# =====================================================
# 2) Neural Network Model (using Flax)
# =====================================================
class MLP(nn.Module):
    """
    Multi-Layer Perceptron using Flax (JAX's neural network library).
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
            x = nn.Dropout(rate=self.dropout, deterministic=not training)(x)
        
        x = nn.Dense(self.output_dim)(x)
        return x


# =====================================================
# 3) Data Pipeline
# =====================================================
def make_loader(batch_size: int, key: jax.random.PRNGKey):
    """
    Load MNIST dataset and create batches.
    """
    (x_train, y_train), _ = mnist.load_data()
    
    # Normalize
    x_train = x_train.astype(np.float32) / 255.0
    y_train = y_train.astype(np.int32)
    
    # Shuffle
    n = len(x_train)
    indices = np.arange(n)
    key, subkey = random.split(key)
    indices = random.permutation(subkey, indices)
    
    x_train = x_train[indices]
    y_train = y_train[indices]
    
    # Create batches
    num_batches = n // batch_size
    x_batches = x_train[:num_batches * batch_size].reshape(
        num_batches, batch_size, 28, 28
    )
    y_batches = y_train[:num_batches * batch_size].reshape(
        num_batches, batch_size
    )
    
    return x_batches, y_batches, key


# =====================================================
# 4) Training Step
# =====================================================
def loss_fn(params, model, x, y):
    """
    Compute cross-entropy loss.
    """
    logits = model.apply({"params": params}, x, training=True)
    
    # One-hot encode labels
    y_onehot = jax.nn.one_hot(y, num_classes=10)
    
    # Compute loss
    loss = jnp.mean(
        optax.softmax_cross_entropy(logits, y_onehot)
    )
    return loss


def accuracy(params, model, x, y):
    """
    Compute accuracy.
    """
    logits = model.apply({"params": params}, x, training=False)
    pred = jnp.argmax(logits, axis=1)
    return jnp.mean(pred == y)


def train_step(state, model, x, y):
    """
    Single training step with gradient update.
    """
    loss_value, grads = jax.value_and_grad(loss_fn)(
        state.params, model, x, y
    )
    new_state = state.apply_gradients(grads=grads)
    acc = accuracy(state.params, model, x, y)
    return new_state, loss_value, acc


# =====================================================
# 5) One Full Experiment Run
# =====================================================
def run_experiment(cfg: Dict, run_id: int, seed: int):
    """
    Run a full training experiment.
    """
    key = seed_everything(seed)
    device = get_device()
    
    # Create model
    model = MLP(
        input_dim=cfg["model"]["input_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        output_dim=cfg["model"]["output_dim"],
        dropout=cfg["model"]["dropout"],
    )
    
    # Initialize model parameters
    key, subkey = random.split(key)
    dummy_input = jnp.ones((1, 28, 28), dtype=jnp.float32)
    params = model.init(subkey, dummy_input, training=False)["params"]
    
    # Create optimizer and training state
    tx = optax.adam(learning_rate=cfg["training"]["learning_rate"])
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx
    )
    
    # Load data
    x_batches, y_batches, key = make_loader(
        cfg["dataset"]["batch_size"], key
    )
    
    num_batches = len(x_batches)
    steps_per_epoch = num_batches
    total_steps = cfg["training"]["epochs"] * steps_per_epoch
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
            
            state, loss_value, acc = train_step(state, model, x, y)
            
            log.append({
                "step": step,
                "loss": float(loss_value),
                "acc": float(acc)
            })
            
            if step == mid_step:
                # Save checkpoint at midpoint
                checkpoint = {"params": state.params}
                with open(
                    f"{cfg['paths']['checkpoints_dir']}/run{run_id}_mid.npy", 
                    "wb"
                ) as f:
                    np.save(f, checkpoint["params"])
    
    # Save final checkpoint
    final_checkpoint = {"params": state.params}
    with open(
        f"{cfg['paths']['checkpoints_dir']}/run{run_id}_final.npy",
        "wb"
    ) as f:
        np.save(f, final_checkpoint["params"])
    
    # Cleanup
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
        
        # Emergency cleanup between runs
        cleanup()
        
        # Save run log
        out = os.path.join(cfg["paths"]["logs_dir"], f"run{run_id}.log")
        with open(out, "w") as f:
            f.write("step,loss,acc\n")
            for row in log:
                f.write(f"{row['step']},{row['loss']},{row['acc']}\n")
        
        print(f"  Saved to {out}\n")
    
    # Aggregate results
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