import torch
import polars as pl
from torch.utils.data import DataLoader

from src.data.seqdataset import SequenceDataset, collate_fn
from src.utils.metrics import Metrics_k

K = 10
MODE = "val"  # переключите на "test" для второй проверки

top_k_items = (
    pl.scan_parquet("dataset/processed/full_data.parquet")
    .sort(["user_id", "timestamp"], descending=[False, True])
    .with_columns(pl.int_range(0, pl.len()).over("user_id").alias("idx"))
    .filter(pl.col("idx") >= 2)  # без двух последних - чтобы не было leakage
    .group_by("item_id").len()
    .sort("len", descending=True)
    .limit(K)
).collect()["item_id"].to_torch()


def last_item_pred(item_seq: torch.Tensor, top_k_items: torch.Tensor, k: int):
    """Тривиальный прогноз: последний item из истории + топ-популярные на добивку."""
    B = item_seq.size(0)
    device = item_seq.device
    preds = []

    for b in range(B):
        items = item_seq[b]
        items = items[items != 0]  # убрать паддинг

        if len(items) == 0:
            preds.append(top_k_items[:k].to(device))
            continue

        last_item = items[-1].item()
        chosen = [last_item]
        for it in top_k_items.tolist():
            if it not in chosen:
                chosen.append(it)
            if len(chosen) == k:
                break

        preds.append(torch.tensor(chosen, device=device, dtype=torch.long))

    return torch.stack(preds)


df = pl.read_parquet("dataset/processed/sequences.parquet")
dataset = SequenceDataset(df, mode=MODE)
loader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=collate_fn)

metrics = Metrics_k(K)
n_repeat, n_total = 0, 0

with torch.no_grad():
    for batch in loader:
        item_seq = batch["item_seq"]
        target = batch["target"]

        for b in range(item_seq.size(0)):
            items = item_seq[b]
            items = items[items != 0]
            n_total += 1
            if len(items) > 0 and items[-1].item() == target[b].item():
                n_repeat += 1

        pred = last_item_pred(item_seq, top_k_items, K)
        metrics.update(target, pred)

answ = metrics.compute()

print(f"Режим: {MODE}")
print(f"Доля таргетов = последнему item в истории (repeat rate): {n_repeat / n_total:.4f} ({n_repeat}/{n_total})")
print(
    f"[copy-last baseline] Hitrate@{K}: {answ['hitrate']:.4f}"
    f" | MRR@{K}: {answ['mrr']:.4f}"
    f" | NDCG@{K}: {answ['ndcg']:.4f}"
)