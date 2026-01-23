# ===============================
# Imports
# ===============================
# os: file/folder handling
# random: Python RNG
# yaml: read config.yaml
# time: optional timing/debug
import os, random, yaml, time

# numpy: numerical arrays + statistics
import numpy as np

# torch core
import torch
import torch.nn as nn
import torch.optim as optim

# torchvision: datasets + transforms
from torchvision import datasets, transforms

# DataLoader: batching + shuffling
from torch.utils.data import DataLoader


# =====================================================
# 1) Reproducibility Utilities
# =====================================================
def seed_everything(seed: int, deterministic: bool = True):
    """
    Forces all major RNG sources to use the same seed.

    This ensures:
      - same weight initialization
      - same DataLoader shuffling
      - same dropout masks
      - same GPU math order

    Without this, two runs of the same script can
    produce different curves.
    """
    random.seed(seed)        # Python random
    np.random.seed(seed)    # NumPy random
    torch.manual_seed(seed) # PyTorch CPU RNG

    # Safe even if CUDA is not available
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # cuDNN settings: slower but reproducible
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Enforce deterministic algorithms (new PyTorch)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def get_device():
    """
    Returns the best available compute device:
    CUDA GPU > Apple MPS > CPU
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# =====================================================
# 2) Neural Network Model
# =====================================================
class MLP(nn.Module):
    """
    Simple 2-layer Multilayer Perceptron.

    Input:  28x28 MNIST image -> 784 vector
    Hidden: ReLU layer
    Output: 10 class logits
    """

    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10, dropout=0.0):
        super().__init__()

        # Build layer list
        layers = [
            nn.Linear(input_dim, hidden_dim),  # Fully connected
            nn.ReLU(),                          # Nonlinearity
        ]

        # Optional dropout for regularization
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Final classifier layer
        layers.append(nn.Linear(hidden_dim, output_dim))

        # nn.Sequential chains layers together
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the network.

        Input x: shape [B, 1, 28, 28]
        Flatten to: [B, 784]
        """
        x = x.view(x.size(0), -1)
        return self.net(x)


# =====================================================
# 3) Data Pipeline
# =====================================================
def make_loader(batch_size: int, seed: int):
    """
    Builds a deterministic DataLoader for MNIST.
    """

    # Convert PIL image -> torch tensor [0,1]
    tfm = transforms.ToTensor()

    # Download MNIST if not present
    train_ds = datasets.MNIST(
        "./data", train=True, download=True, transform=tfm
    )

    # Generator with fixed seed for reproducible shuffling
    g = torch.Generator().manual_seed(seed)

    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=g
    )
    return loader


# =====================================================
# 4) Training for One Epoch
# =====================================================
def train_one_epoch(model, loader, opt, loss_fn, device):
    """
    Trains the model for one full pass over the dataset.
    Returns average loss and accuracy.
    """
    model.train()  # enable training mode

    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in loader:
        # Move batch to GPU/CPU
        x, y = x.to(device), y.to(device)

        # Forward pass
        logits = model(x)

        # Compute loss
        loss = loss_fn(logits, y)

        # Backpropagation
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Metrics
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    # Return averages
    return total_loss / total, total_correct / total


# =====================================================
# 5) One Full Experiment Run
# =====================================================
def run_experiment(cfg, run_id: int, seed: int):
    """
    Runs training for one seed and logs every step.
    """

    # Fix randomness
    seed_everything(seed, deterministic=cfg["repro"]["deterministic"])
    device = get_device()

    loader = make_loader(cfg["dataset"]["batch_size"], seed)

    model = MLP(**cfg["model"]).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()

    # Create folders
    os.makedirs(cfg["paths"]["checkpoints_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["logs_dir"], exist_ok=True)

    steps_per_epoch = len(loader)
    total_steps = cfg["training"]["epochs"] * steps_per_epoch
    mid_step = total_steps // 2
    step = 0

    log = []  # list of {step, loss, acc}

    for epoch in range(cfg["training"]["epochs"]):
        for x, y in loader:
            step += 1
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = loss_fn(logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()

            acc = (logits.argmax(1) == y).float().mean().item()
            log.append({"step": step, "loss": loss.item(), "acc": acc})

            # Midpoint checkpoint
            if step == mid_step:
                torch.save(
                    model.state_dict(),
                    f"{cfg['paths']['checkpoints_dir']}/run{run_id}_mid.pt"
                )

    # Final checkpoint
    torch.save(
        model.state_dict(),
        f"{cfg['paths']['checkpoints_dir']}/run{run_id}_final.pt"
    )

    return log


# =====================================================
# 6) Entry Point
# =====================================================
def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    all_runs = []

    for i in range(cfg["num_runs"]):
        run_id = i + 1
        seed = cfg["seed_base"] + i
        print(f"[Run {run_id}] seed={seed}")

        log = run_experiment(cfg, run_id, seed)
        all_runs.append(log)

        # Save per-run log
        out = os.path.join(cfg["paths"]["logs_dir"], f"run{run_id}.log")
        with open(out, "w") as f:
            f.write("step,loss,acc\n")
            for row in log:
                f.write(f"{row['step']},{row['loss']},{row['acc']}\n")

    # Convert to numpy array: [runs, steps, 2]
    steps = len(all_runs[0])
    arr = np.zeros((len(all_runs), steps, 2), dtype=np.float32)

    for r in range(len(all_runs)):
        for t in range(steps):
            arr[r, t, 0] = all_runs[r][t]["loss"]
            arr[r, t, 1] = all_runs[r][t]["acc"]

    np.save(os.path.join(cfg["paths"]["logs_dir"], "all_runs.npy"), arr)


if __name__ == "__main__":
    main()