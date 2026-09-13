"""Mixed-precision experiments (handout §2.1.5).

    uv run python -m cs336_systems.mixed_precision accumulation
    uv run python -m cs336_systems.mixed_precision toy --dtype fp16   # dtypes inside autocast
"""

from __future__ import annotations

import argparse

import torch
from torch import nn

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


def accumulation_experiment() -> dict[str, float]:
    """The four accumulation loops from problem ``mixed_precision_accumulation``."""
    results = {}
    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float32)
    results["fp32 accumulator, fp32 addend"] = s.item()

    s = torch.tensor(0, dtype=torch.float16)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    results["fp16 accumulator, fp16 addend"] = s.item()

    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    results["fp32 accumulator, fp16 addend (implicit upcast)"] = s.item()

    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        x = torch.tensor(0.01, dtype=torch.float16)
        s += x.type(torch.float32)
    results["fp32 accumulator, fp16 addend cast to fp32"] = s.item()
    return results


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x


def toy_autocast_dtypes(device: str, dtype: torch.dtype) -> dict[str, str]:
    """Report the dtype of every intermediate of ``ToyModel`` under autocast."""
    torch.manual_seed(0)
    model = ToyModel(8, 4).to(device)
    x = torch.randn(16, 8, device=device)
    y = torch.randint(0, 4, (16,), device=device)
    seen: dict[str, str] = {}

    def record(name):
        def hook(mod, inp, out):
            seen[name] = str(out.dtype)

        return hook

    model.fc1.register_forward_hook(record("fc1 output"))
    model.ln.register_forward_hook(record("layernorm output"))
    with torch.autocast(device_type=device, dtype=dtype):
        seen["fc1.weight inside autocast"] = str(model.fc1.weight.dtype)
        logits = model(x)
        seen["logits"] = str(logits.dtype)
        loss = nn.functional.cross_entropy(logits, y)
        seen["loss"] = str(loss.dtype)
    loss.backward()
    seen["fc1.weight.grad"] = str(model.fc1.weight.grad.dtype)
    seen["ln.weight.grad"] = str(model.ln.weight.grad.dtype)
    return seen


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("experiment", choices=["accumulation", "toy"])
    p.add_argument("--dtype", default="fp16", choices=sorted(DTYPES))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.experiment == "accumulation":
        for k, v in accumulation_experiment().items():
            print(f"{k:50s} {v:.6f}")
        print(f"{'exact':50s} 10.000000")
    else:
        for k, v in toy_autocast_dtypes(args.device, DTYPES[args.dtype]).items():
            print(f"{k:30s} {v}")


if __name__ == "__main__":
    main()
