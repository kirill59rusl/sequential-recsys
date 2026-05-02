from torch.utils.data import DataLoader
import torch
from src.data.seqdataset import SequenceDataset
from src.data.seqdataset import collate_fn
import polars as pl
from src.utils.metrics import Metrics_k

LOADER_BATCH=32
K=5
EVENT_WEIGHTS={0:1,1:3,2:5} # view - 1 ; addtocart - 3 ; view - 5

top_k_items=(pl.scan_parquet("dataset/processed/full_data.parquet")
             .sort(['user_id','timestamp'],descending=[False,True])
             .with_columns(
                pl.int_range(0, pl.len()).over("user_id").alias("idx")
             )
             .filter((pl.col("idx")>=2)) #отсекаем два последних - чтобы не было leakage
             .group_by("item_id").len()
             .sort("len",descending=True).limit(K)).collect()["item_id"].to_torch()


dataset=pl.read_parquet("dataset/processed/sequences.parquet")

datatest=SequenceDataset(dataset, mode='val')

loadertest = DataLoader(
    datatest,
    batch_size=LOADER_BATCH,
    shuffle=False,
    collate_fn=collate_fn
)

def weighted_topk(
    item_seq: torch.Tensor,
    event_seq: torch.Tensor,
    top_k_items: torch.Tensor,
    k: int,
):
    """
    item_seq  : [B, L]
    event_seq : [B, L]
    top_k_items : [M]
    return -> [B, K]
    """

    device = item_seq.device
    B = item_seq.size(0)

    # веса действий -> tensor lookup
    max_event = max(EVENT_WEIGHTS.keys())
    weight_lookup = torch.zeros(max_event + 1, device=device)

    for ev, w in EVENT_WEIGHTS.items():
        weight_lookup[ev] = w

    preds = []

    for b in range(B):
        items = item_seq[b]
        events = event_seq[b]

        # убрать padding
        mask = items != 0
        items = items[mask]
        events = events[mask]

        # пустая история
        if len(items) == 0:
            preds.append(top_k_items[:k].to(device))
            continue

        weights = weight_lookup[events]

        # суммируем веса одинаковых item_id
        uniq_items, inverse = torch.unique(items, return_inverse=True)

        scores = torch.zeros(
            len(uniq_items),
            dtype=torch.float32,
            device=device,
        )

        scores.scatter_add_(0, inverse, weights)

        # сортировка по score desc
        order = torch.argsort(scores, descending=True)
        ranked_items = uniq_items[order]

        # если уже хватает
        if len(ranked_items) >= k:
            preds.append(ranked_items[:k])
            continue

        # дополняем глобальным top-k без повторов
        chosen = ranked_items.tolist()
        chosen_set = set(chosen)

        for item in top_k_items.tolist():
            if item not in chosen_set:
                chosen.append(item)
                chosen_set.add(item)

            if len(chosen) == k:
                break

        preds.append(
            torch.tensor(chosen, device=device, dtype=torch.long)
        )

    return torch.stack(preds)

metrics=Metrics_k(K)

with torch.no_grad():
    for batch in loadertest:
        target=batch['target']
        item_seq=batch['item_seq']
        event_seq=batch['events_seq']
        pred=weighted_topk(item_seq,event_seq,top_k_items,K)
        metrics.update(target,pred)

answ=metrics.compute()

print(f"Hitrate@{K}: {answ['hitrate']} | MRR@{K}: {answ['mrr']} | NDCG@{K}: {answ['ndcg']}")





