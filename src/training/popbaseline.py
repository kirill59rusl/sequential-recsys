from torch.utils.data import DataLoader
import torch
from src.data.seqdataset import SequenceDataset
from src.data.seqdataset import collate_fn
import polars as pl
from src.utils.metrics import Metrics_k

#baseline - popularity, берём k самых популярных предметов из истории
K=10
LOADER_BATCH=32 

top_k_items=(pl.scan_parquet("dataset/processed/full_data.parquet")
             .sort(['user_id','timestamp'],descending=[False,True])
             .with_columns(
                pl.int_range(0, pl.len()).over("user_id").alias("idx")
             )
             .filter((pl.col("idx")!=1) & (pl.col("idx")!=0)) #отсекаем два последних - чтобы не было leakage
             .group_by("item_id").len()
             .sort("len",descending=True).limit(K)).collect()["item_id"].to_torch()

dataset=pl.read_parquet("dataset/processed/sequences.parquet")

datatest=SequenceDataset(dataset, mode='test')

loadertest = DataLoader(
    datatest,
    batch_size=LOADER_BATCH,
    shuffle=False,
    collate_fn=collate_fn
)


metrics=Metrics_k(K)

with torch.no_grad():
    for batch in loadertest:
        target=batch['target']
        b=target.size(0)
        pred=top_k_items.expand(b,-1)
        metrics.update(target,pred)

        
answ=metrics.compute()


print(f"Hitrate@10: {answ['hitrate']} | MRR@10: {answ['mrr']} | NDCG@10: {answ['ndcg']}")