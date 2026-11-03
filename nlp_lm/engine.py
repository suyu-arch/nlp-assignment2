import math
import random
import time

import torch
from torch.nn import functional as F


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def choose_device(requested="auto"):
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable in this Python environment; use cpu or auto.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this Python environment; use cpu or auto.")
    return torch.device(requested)


def token_loss(logits, targets):
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
                           ignore_index=0, reduction="sum")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def run_epoch(model, loader, device, optimizer=None, max_batches=None, log_every=0):
    training = optimizer is not None
    model.train(training)
    total_nll, total_tokens = 0.0, 0
    synchronize(device)
    started = time.perf_counter()
    with torch.set_grad_enabled(training):
        for step, (inputs, targets) in enumerate(loader, 1):
            inputs, targets = inputs.to(device), targets.to(device)
            count = int(targets.ne(0).sum().item())
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss_sum = token_loss(model(inputs), targets)
            loss = loss_sum / count
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; check the learning rate and data.")
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_nll += loss_sum.item()
            total_tokens += count
            if log_every and step % log_every == 0:
                print(f"  batch {step}/{len(loader)} | token loss {total_nll / total_tokens:.4f}", flush=True)
            if max_batches is not None and step >= max_batches:
                break
    synchronize(device)
    seconds = time.perf_counter() - started
    if not total_tokens:
        raise ValueError("No target tokens were evaluated.")
    mean_nll = total_nll / total_tokens
    return {"loss": mean_nll, "ppl": math.exp(mean_nll) if mean_nll < 700 else float("inf"),
            "tokens": total_tokens, "seconds": seconds}
