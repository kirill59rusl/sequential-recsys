import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.utils.metrics import Metrics_k


class NegativeSampler:
    
    def __init__(self, num_items):
        self.num_items = num_items

    def sample(self, positives, num_negatives=1):
        B = positives.size(0)
        return torch.randint(
            1, self.num_items + 1, (B, num_negatives), device=positives.device
        )


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch in tqdm(loader, leave=False, desc="train"):
        item_seq = batch["item_seq"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()
        logits = model(item_seq, mask)[:, -1, :]
        loss = F.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device, k=10):
    model.eval()
    metrics = Metrics_k(k)
    total_loss = 0.0

    for batch in tqdm(loader, leave=False, desc="eval"):
        item_seq = batch["item_seq"].to(device)
        target = batch["target"].to(device)
        mask = batch["mask"].to(device)

        logits = model(item_seq, mask)[:, -1, :]
        loss = F.cross_entropy(logits, target)

        total_loss += loss.item()
        metrics.update(target, logits)

    scores = metrics.compute()
    scores["loss"] = total_loss / len(loader)
    return scores