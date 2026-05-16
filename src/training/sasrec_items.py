import numpy as np
import torch
import polars as pl
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from src.models.sasrec import SASRec
from src.data.seqdataset import SequenceDataset
from src.data.seqdataset import collate_fn
from src.utils.metrics import Metrics_k

class NegativeSampler:
    def __init__(self, num_items):
        self.num_items = num_items

    def sample(self, positives, num_negatives=1):

        B = positives.size(0)

        negatives = torch.randint(
            1,
            self.num_items + 1,
            (B, num_negatives),
            device=positives.device
        )

        return negatives


def train_epoch(model, loader, optimizer, device):

    model.train()

    total_loss = 0

    for batch in tqdm(loader):

        item_seq = batch['item_seq'].to(device)
        target = batch['target'].to(device)
        mask = batch['mask'].to(device)

        optimizer.zero_grad()

        logits = model.forward(
            item_seq,
            mask
        )[:,-1,:]

        loss = F.cross_entropy(
            logits,
            target
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    k=10
):

    model.eval()

    metrics = Metrics_k(k)

    total_loss = 0

    for batch in tqdm(loader):

        item_seq = batch['item_seq'].to(device)
        target = batch['target'].to(device)
        mask = batch['mask'].to(device)

        logits = model.forward(
            item_seq,
            mask
        )[:,-1,:]

        loss = torch.nn.functional.cross_entropy(
            logits,
            target
        )

        total_loss += loss.item()

        metrics.update(
            target,
            logits
        )

    scores = metrics.compute()

    scores['loss'] = total_loss / len(loader)

    return scores


df=pl.read_parquet("dataset/processed/sequences.parquet")

num_items = int(df.select(pl.col('item_sequence').list.max().max()).item()) + 1
print(f"num_items: {num_items}")

class args:
    num_items = num_items
    max_len = 50
    hidden_dim = 128
    num_blocks = 2
    num_heads = 2
    num_layers = 2
    dropout = 0.2
    batch_size = 128
    lr = 1e-3
    num_epochs = 10
    device='cpu'
        
        
model = SASRec(args).to(args.device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr,
    weight_decay=1e-4
)

train_dataset = SequenceDataset(
    df,
    mode='train'
)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    collate_fn=collate_fn
)

val_dataset = SequenceDataset(
    df,
    mode='val'
)

val_loader = DataLoader(
    val_dataset,
    batch_size=256,
    shuffle=False,
    collate_fn=collate_fn
)

for epoch in range(args.num_epochs):

    train_loss = train_epoch(
        model,
        train_loader,
        optimizer,
        args.device
    )

    val_scores=evaluate(
        model,
        val_loader,
        args.device
    )

    print(
        f'Epoch {epoch}'
        f' | train_loss={train_loss:.4f}'
        f' | val_loss={val_scores["loss"]:.4f}'
        f' | HR@10={val_scores["hitrate"]:.4f}'
        f' | MRR@10={val_scores["mrr"]:.4f}'
        f' | NDCG@10={val_scores["ndcg"]:.4f}'
    )
