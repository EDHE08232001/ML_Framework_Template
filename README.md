# PyTorch vs JAX: Template Comparison Guide

## Overview
This document explains the key differences between the PyTorch template and the JAX template, useful for understanding two different paradigms in deep learning.

---

## 1. **Paradigm Difference**

### PyTorch
- **Imperative & Stateful**: Models are classes with mutable state
- **Object-Oriented**: Neural networks are objects with `forward()` methods
- **Eager Execution**: Operations execute immediately

### JAX
- **Functional & Immutable**: Pure functions, no mutable state
- **Functional Programming**: Models composed as functions
- **JIT Compilation**: Can compile to static graphs for performance
- **Composable Transformations**: `grad`, `vmap`, `pmap` for derivatives, vectorization

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
- JAX uses `@nn.compact` decorator (modern Flax pattern)
- No explicit layer storage; layers are functional
- Training mode is a parameter, not layer state

---

## 3. **Initialization**

### PyTorch
```python
model = MLP(input_dim=784, hidden_dim=256, output_dim=10, dropout=0.0)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
```

### JAX
```python
model = MLP(input_dim=784, hidden_dim=256, output_dim=10, dropout=0.0)
key = random.key(seed)
dummy_input = jnp.ones((1, 28, 28), dtype=jnp.float32)
params = model.init(key, dummy_input, training=False)["params"]

tx = optax.adam(learning_rate=0.001)
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
```

**Key differences:**
- JAX requires explicit parameter initialization with dummy input
- Parameters are separated from model architecture
- Optimizer is created independently via `optax`
- `TrainState` bundles optimizer state and parameters

---

## 4. **Random Number Generation**

### PyTorch
```python
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
```

### JAX
```python
key = random.key(seed)
key, subkey = random.split(key)
# Use subkey for operations, keep key for next iteration
```

**Key differences:**
- JAX uses explicit key passing (functional approach)
- Must split keys for each random operation
- No global random state

---

## 5. **Training Loop**

### PyTorch (stateful, imperative)
```python
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### JAX (functional, pure)
```python
def train_step(state, model, x, y):
    loss_value, grads = jax.value_and_grad(loss_fn)(state.params, model, x, y)
    new_state = state.apply_gradients(grads=grads)
    acc = accuracy(state.params, model, x, y)
    return new_state, loss_value, acc

def loss_fn(params, model, x, y):
    logits = model.apply({"params": params}, x, training=True)
    y_onehot = jax.nn.one_hot(y, num_classes=10)
    loss = jnp.mean(optax.softmax_cross_entropy(logits, y_onehot))
    return loss
```

**Key differences:**
- JAX: loss function is explicit function of parameters
- JAX uses `jax.value_and_grad()` to get loss AND gradients
- PyTorch: backward pass modifies optimizer state in-place
- JAX: returns new state (immutable)
- No `zero_grad()` needed in JAX

---

## 6. **JIT Compilation (JAX advantage)**

### Pure JAX
```python
@jax.jit
def train_step(state, model, x, y):
    loss_value, grads = jax.value_and_grad(loss_fn)(state.params, model, x, y)
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss_value

# Can batch vectorize
train_step_batched = jax.vmap(train_step, in_axes=(None, None, 0, 0))
```

**JAX can:**
- JIT compile pure functions for speed
- Automatically vectorize with `vmap`
- Parallelize with `pmap`
- Differentiate through differentiation (`grad(grad(...))`)

---

## 7. **Device Handling**

### PyTorch
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
x = x.to(device)
```

### JAX
```python
device = jax.devices()[0].platform
# JAX automatically uses available accelerators
# No manual device placement needed
```

**Key differences:**
- JAX handles devices implicitly
- Arrays stay on current device automatically
- More transparent to user

---

## 8. **Checkpointing**

### PyTorch
```python
torch.save(model.state_dict(), "model.pt")
model.load_state_dict(torch.load("model.pt"))
```

### JAX
```python
# Save parameters as numpy
np.save("params.npy", params)

# Load parameters
params = np.load("params.npy", allow_pickle=True).item()
```

**Key differences:**
- JAX checkpoints are just NumPy arrays (simpler, more portable)
- No special serialization framework needed
- Explicit parameter passing

---

## 9. **Pros & Cons Summary**

### PyTorch Pros
✅ Intuitive imperative syntax
✅ Easy to debug (eager execution)
✅ Large ecosystem, many pretrained models
✅ Great for research & prototyping
✅ Good documentation

### PyTorch Cons
❌ Slower at scale without careful optimization
❌ Harder to parallelize across devices
❌ Mutation makes parallelization tricky

### JAX Pros
✅ Composable transformations (grad, vmap, pmap)
✅ JIT compilation for speed
✅ Natural expression of math (NumPy-like)
✅ Better for distributed computing
✅ Pure functions are easier to reason about

### JAX Cons
❌ Steeper learning curve (functional paradigm)
❌ Key-based randomness requires discipline
❌ Smaller ecosystem
❌ Harder to debug (graph compilation)
❌ More boilerplate code

---

## 10. **Educational Takeaways**

1. **Different Paradigms**: PyTorch is imperative; JAX is functional
2. **State Management**: PyTorch mutates; JAX returns new values
3. **Randomness**: JAX's explicit key passing prevents bugs
4. **Composability**: JAX's transformations enable automatic differentiation, vectorization, parallelization
5. **Speed**: JAX's JIT can be faster, but requires pure functions
6. **Tradeoff**: JAX trades ease-of-use for power and composability

---

## Quick Command Reference

| Task | PyTorch | JAX |
|------|---------|-----|
| Seed | `torch.manual_seed(42)` | `key = random.key(42)` |
| Model init | `model = MLP(...)` | `params = model.init(...)` |
| Forward pass | `logits = model(x)` | `logits = model.apply({"params": params}, x)` |
| Gradients | `loss.backward()` | `grads = grad(loss_fn)(params)` |
| Optimizer step | `opt.step()` | `state = state.apply_gradients(grads)` |
| JIT | N/A (eager) | `@jax.jit` decorator |
| Vectorize | Manual batching | `jax.vmap(fn)` |
| Save | `torch.save()` | `np.save()` |

---

## Running the Templates

### PyTorch
```bash
pip install -r requirements.txt
python pytorch_template.py
```

### JAX
```bash
pip install -r jax_requirements.txt
python jax_template.py
```

Both should produce identical experiment structures (multiple runs, seeded for reproducibility, saved metrics).