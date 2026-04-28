from torch.utils.data import DataLoader
import torch
from src.data.seqdataset import SequenceDataset
from src.data.seqdataset import collate_fn
import polars as pl


#baseline - popularity, берём k самых популярных предметов из истории
K=10
top_k_items=(pl.scan_parquet("dataset/processed/full_data.parquet")
             .sort(['user_id','timestamp'],descending=[False,True])
             .with_columns(
                pl.int_range(0, pl.len()).over("user_id").alias("idx")
             )
             .filter((pl.col("idx")!=1) & (pl.col("idx")!=2))
             .group_by("item_id").len()
             .sort("len",descending=True).limit(K)).collect()["item_id"].to_torch()

dataset=pl.read_parquet("dataset/processed/sequences.parquet")

LOADER_BATCH=32

datatest=SequenceDataset(dataset, mode='test')

loadertest = DataLoader(
    datatest,
    batch_size=LOADER_BATCH,
    shuffle=False,
    collate_fn=collate_fn
)

def recall_k(target:torch.Tensor,pred:torch.Tensor):  #вынести в модуль
    b_size=target.size(0)
    answ=pred.expand(b_size,-1)
    
    return (target.unsqueeze(1)==answ).any(dim=1).float().mean()

recall=0
n_samples=0
with torch.no_grad():
    for batch in loadertest:
        y=batch['target']
        b=y.size(0)
        recall+=recall_k(y,top_k_items)*b
        n_samples+=b

print(f"Recall@10: {recall/n_samples}")