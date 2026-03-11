# PyTorch vs JAX: Template Comparison Guide

## Overview
This document breaks down the key architectural differences between PyTorch and JAX. Building a strong intuition for both imperative (PyTorch) and functional (JAX) paradigms is a massive asset for technical interviews and gives you the flexibility to choose the right tool for different machine learning pipelines.

---

## 1. **Paradigm Difference**

### PyTorch
- **Imperative & Stateful**: Models are classes with mutable state.
- **Object-Oriented**: Neural networks are objects with `forward()` methods.
- **Eager Execution**: Operations execute immediately line-by-line (highly debuggable).

### JAX
- **Functional & Immutable**: Pure functions, absolutely no mutable state.
- **Functional Programming**: Models and training steps are composed as pure functions.
- **JIT Compilation**: Compiles your Python functions into highly optimized static XLA graphs.
- **Composable Transformations**: `grad` (derivatives), `vmap` (vectorization), `pmap` (parallelization).

---

## 2. **Model Definition**

### PyTorch
```python
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        if self.dropout.p > 0:
            x = self.dropout(x)
        return self.fc2(x)

```

### JAX (with Flax)

```python
class MLP(nn.Module):
    input_dim: int = 784
    hidden_dim: int = 256
    output_dim: int = 10
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, training: bool = False):
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        if self.dropout > 0.0:
            x = nn.Dropout(rate=self.dropout, deterministic=not training)(x)
        x = nn.Dense(self.output_dim)(x)
        return x

```

**Key differences:**

* JAX/Flax uses the `@nn.compact` decorator for inline submodule definition.
* Layers in JAX don't hold their own weights; they are purely mathematical transformations.

---

## 3. **Initialization & State**

### PyTorch

```python
model = MLP(input_dim=784, hidden_dim=256, output_dim=10, dropout=0.0).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

```

### JAX

```python
model = MLP(input_dim=784, hidden_dim=256, output_dim=10, dropout=0.0)

# 1. Initialize variables using a PRNG key and a dynamically sized dummy input
dummy_input = jnp.ones((batch_size, 28, 28), dtype=jnp.float32)
variables = model.init({"params": init_key, "dropout": drop_key}, dummy_input, training=False)

# 2. Bundle params and optimizer into an immutable TrainState
tx = optax.adam(learning_rate=0.001)
state = train_state.TrainState.create(apply_fn=model.apply, params=variables["params"], tx=tx)

```

**Key differences:**

* PyTorch initializes weights under the hood during class instantiation.
* JAX requires an explicit dummy input (sized correctly to your batch) to trace the graph and generate parameter shapes.
* JAX explicitly separates the model architecture from the parameter state.

---

## 4. **Random Number Generation**

### PyTorch

```python
torch.manual_seed(42) # Global state mutated

```

### JAX

```python
key = random.key(42)
key, subkey = random.split(key) # Explicit functional splitting

```

**Key differences:**

* JAX has **no global random state**. You must explicitly pass and split PRNG keys for *every* random operation. For example, a new key must be split and passed to the model on every single training step to ensure Dropout masks are unique.

---

## 5. **Training Loop & JIT Compilation**

### PyTorch (Stateful, Imperative)

```python
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train() # Set state
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        
        optimizer.zero_grad()
        loss.backward() # Mutates gradients globally
        optimizer.step() # Mutates weights

```

### JAX (Functional, Compiled)

```python
@jax.jit
def train_step(state, x, y, dropout_key):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, x, training=True, rngs={"dropout": dropout_key})
        y_onehot = jax.nn.one_hot(y, num_classes=10)
        return jnp.mean(optax.softmax_cross_entropy(logits, y_onehot)), logits

    # Get loss AND gradients simultaneously
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    
    # Return an entirely new state object
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

```

**Key differences:**

* The `@jax.jit` decorator compiles the entire `train_step` into an ultra-fast XLA executable.
* PyTorch modifies the optimizer state in-place; JAX returns a mathematically new `TrainState`.
* No `zero_grad()` is needed in JAX because gradients are calculated functionally as outputs.

---

## 6. **Device Handling (MPS & CUDA)**

Both frameworks are configured in these templates to automatically detect your hardware.

This means you can confidently prototype locally on Apple Silicon (using MPS for Mac) and seamlessly deploy the exact same scripts to CUDA-enabled workstations or gaming laptops for heavy training runs without modifying a single line of device-placement code.

### PyTorch

```python
# Custom fallback required
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)
x = x.to(device)

```

### JAX

```python
# Fully automatic
device = jax.devices()[0].platform 
# JAX automatically places arrays on the fastest available accelerator

```

---

## 7. **Pros & Cons Summary**

### PyTorch Pros

✅ Intuitive, Pythonic syntax (great for quick debugging).
✅ Massive ecosystem and industry-standard for most research.
✅ Eager execution makes printing tensor shapes mid-forward pass trivial.

### PyTorch Cons

❌ Eager mode has python overhead (though `torch.compile` is bridging this gap).
❌ Mutating state makes complex parallelization trickier.

### JAX Pros

✅ First-class composable transformations (`vmap` is magic for custom training loops).
✅ Extremely fast once compiled.
✅ Functional purity makes massive distributed computing much safer.

### JAX Cons

❌ Steep learning curve.
❌ Key-based randomness requires rigorous discipline (easy to accidentally reuse a dropout mask).
❌ Harder to debug (you cannot just throw a `print()` inside a JIT-compiled function).

---

## Quick Command Reference

| Task | PyTorch | JAX |
| --- | --- | --- |
| Seed | `torch.manual_seed(42)` | `key = random.key(42)` |
| Model init | `model = MLP(...)` | `variables = model.init(...)` |
| Forward pass | `logits = model(x)` | `logits = state.apply_fn({"params": params}, x)` |
| Gradients | `loss.backward()` | `grads = jax.grad(loss_fn)(params)` |
| Optimizer step | `opt.step()` | `state = state.apply_gradients(grads)` |
| JIT | `torch.compile(model)` | `@jax.jit` decorator |
| Vectorize | Manual batching | `jax.vmap(fn)` |

---

## Running the Templates

Ensure you have your virtual environment active and dependencies installed from the respective `requirements.txt` files.

### PyTorch

```bash
python pytorch_template.py

```

### JAX

```bash
python jax_template.py

```
