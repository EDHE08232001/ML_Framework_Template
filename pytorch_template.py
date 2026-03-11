# ===============================
# Imports
# ===============================
import os, random, yaml, time, gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# =====================================================
# 1) Reproducibility Utilities
# =====================================================
def seed_everything(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# =====================================================
# 1.5) GPU / Memory Cleanup
# =====================================================
def cleanup(model=None, opt=None, loss_fn=None):
    """
    Releases Python + GPU memory.
    Works for CUDA, Apple MPS, and CPU.
    """

    if model is not None:
        del model
    if opt is not None:
        del opt
    if loss_fn is not None:
        del loss_fn

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


# =====================================================
# 2) Neural Network Model
# =====================================================
class MLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10, dropout=0.0):
        super().__init__()

        layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        ]

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)


# =====================================================
# 3) Data Pipeline
# =====================================================
def make_loader(batch_size: int, seed: int):
    tfm = transforms.ToTensor()

    train_ds = datasets.MNIST(
        "./data", train=True, download=True, transform=tfm
    )

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
    model.train()

    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = loss_fn(logits, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return total_loss / total, total_correct / total


# =====================================================
# 5) One Full Experiment Run
# =====================================================
def run_experiment(cfg, run_id: int, seed: int):
    seed_everything(seed, deterministic=cfg["repro"]["deterministic"])
    device = get_device()

    loader = make_loader(cfg["dataset"]["batch_size"], seed)

    model = MLP(**cfg["model"]).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs(cfg["paths"]["checkpoints_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["logs_dir"], exist_ok=True)

    steps_per_epoch = len(loader)
    total_steps = cfg["training"]["epochs"] * steps_per_epoch
    mid_step = total_steps // 2
    step = 0

    log = []

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

            if step == mid_step:
                torch.save(
                    model.state_dict(),
                    f"{cfg['paths']['checkpoints_dir']}/run{run_id}_mid.pt"
                )

    torch.save(
        model.state_dict(),
        f"{cfg['paths']['checkpoints_dir']}/run{run_id}_final.pt"
    )

    # ---- Cleanup GPU / RAM ----
    cleanup(model, opt, loss_fn)

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

        # Emergency cleanup between runs
        cleanup()

        out = os.path.join(cfg["paths"]["logs_dir"], f"run{run_id}.log")
        with open(out, "w") as f:
            f.write("step,loss,acc\n")
            for row in log:
                f.write(f"{row['step']},{row['loss']},{row['acc']}\n")

    steps = len(all_runs[0])
    arr = np.zeros((len(all_runs), steps, 2), dtype=np.float32)

    for r in range(len(all_runs)):
        for t in range(steps):
            arr[r, t, 0] = all_runs[r][t]["loss"]
            arr[r, t, 1] = all_runs[r][t]["acc"]

    np.save(os.path.join(cfg["paths"]["logs_dir"], "all_runs.npy"), arr)


if __name__ == "__main__":
    main()
