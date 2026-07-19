# TT-v1 (Tiki-Taka) on MNIST, paste-ready for the analog-env-tierA colab.
# Mirrors the Analog SGD baseline cell: same net, same data, same epochs.
# Adds tensorboard logging so we can compare runs side by side.
#
# Expected outcome per Wu et al.: TT-v1 recovers most of the ~1.7 pt gap
# between FP SGD (98.1%) and Analog SGD (96.4%).

import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter

from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.optim import AnalogSGD
from aihwkit.simulator.configs import (
    UnitCellRPUConfig,
    TransferCompound,
    SoftBoundsDevice,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 30
BATCH = 64
LR = 0.1
RUN_NAME = "tt_v1"

# ---- Tiki-Taka v1 config ----
# TransferCompound with two SoftBounds devices = TT-v1 (Gokmen & Haensch 2020).
# A array (fast, gets gradient updates) transfers periodically to C array (slow, holds weights).
rpu_config = UnitCellRPUConfig(
    device=TransferCompound(
        unit_cell_devices=[
            SoftBoundsDevice(w_min=-1.0, w_max=1.0),
            SoftBoundsDevice(w_min=-1.0, w_max=1.0),
        ],
        transfer_every=1,
        gamma=0.0,
        scale_transfer_lr=True,
        transfer_lr=1.0,
    )
)

# ---- data (same normalization as baseline cell) ----
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
train_set = datasets.MNIST("data", train=True, download=True, transform=transform)
test_set = datasets.MNIST("data", train=False, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=BATCH, shuffle=True)
test_loader = DataLoader(test_set, batch_size=256)

# ---- model: 3-layer FCN, analog layers ----
model = AnalogSequential(
    nn.Flatten(),
    AnalogLinear(784, 256, rpu_config=rpu_config),
    nn.Sigmoid(),
    AnalogLinear(256, 128, rpu_config=rpu_config),
    nn.Sigmoid(),
    AnalogLinear(128, 10, rpu_config=rpu_config),
    nn.LogSoftmax(dim=1),
).to(DEVICE)

optimizer = AnalogSGD(model.parameters(), lr=LR)
optimizer.regroup_param_groups(model)
criterion = nn.NLLLoss()
writer = SummaryWriter(f"runs/{RUN_NAME}")


def evaluate():
    model.eval()
    correct = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
    return correct / len(test_set)


start = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    train_loss = total_loss / len(train_set)
    acc = evaluate()
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Accuracy/test", acc, epoch)
    print(
        f"Epoch {epoch:2d} - Training loss: {train_loss:.6f}\tTest Accuracy: {acc:.4f}"
    )

writer.close()
print(f"\nTraining Time (s) = {time.time() - start}")
print(f"TT-v1 final test acc: {evaluate()*100:.2f}%  (train loss ~{train_loss:.3f})")
