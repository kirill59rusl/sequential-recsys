import os
import random

import hydra
import numpy as np
import polars as pl
import torch
import wandb
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader

from src.data.seqdataset import SequenceDataset, collate_fn
from src.models.sasrec import SASRec
from src.training.sasrec.engine import train_epoch, evaluate


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@hydra.main(config_path="../../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    device = cfg.device if (cfg.device == "cuda" and torch.cuda.is_available()) else "cpu"

    os.environ["WANDB_MODE"] = cfg.wandb.mode
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        mode=cfg.wandb.mode,
        tags=list(cfg.wandb.tags),
        config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore[arg-type]
    )

    data_path = os.path.join(hydra.utils.get_original_cwd(), cfg.data.sequences_path)
    df = pl.read_parquet(data_path)

    num_items = int(df.select(pl.col("item_sequence").list.max().max()).item()) + 1
    wandb.config.update({"num_items": num_items})
    print(f"num_items: {num_items} | device: {device}")

    with open_dict(cfg.model):
       cfg.model.num_items = num_items
    model = SASRec(cfg.model).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    train_loader = DataLoader(
        SequenceDataset(df, max_len=cfg.model.max_len, mode="train"),
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        SequenceDataset(df, max_len=cfg.model.max_len, mode="val"),
        batch_size=cfg.train.val_batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        SequenceDataset(df, max_len=cfg.model.max_len, mode="test"),
        batch_size=cfg.train.val_batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )

    os.makedirs(cfg.train.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.train.ckpt_dir, "best.pt")

    best_hr, patience = -1.0, 0
    for epoch in range(cfg.train.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_scores = evaluate(model, val_loader, device, k=cfg.train.k)

        print(
            f"Epoch {epoch}"
            f" | train_loss={train_loss:.4f}"
            f" | val_loss={val_scores['loss']:.4f}"
            f" | HR@{cfg.train.k}={val_scores['hitrate']:.4f}"
            f" | MRR@{cfg.train.k}={val_scores['mrr']:.4f}"
            f" | NDCG@{cfg.train.k}={val_scores['ndcg']:.4f}"
        )

        wandb.log(
            {
                "epoch": epoch,
                "train/loss": train_loss,
                "val/loss": val_scores["loss"],
                f"val/hitrate@{cfg.train.k}": val_scores["hitrate"],
                f"val/mrr@{cfg.train.k}": val_scores["mrr"],
                f"val/ndcg@{cfg.train.k}": val_scores["ndcg"],
            }
        )

        if val_scores["hitrate"] > best_hr:
            best_hr, patience = val_scores["hitrate"], 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience += 1
            if patience >= cfg.train.early_stopping_patience:
                print(f"Early stopping на эпохе {epoch}")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_scores = evaluate(model, test_loader, device, k=cfg.train.k)
    print(
        f"[TEST] HR@{cfg.train.k}={test_scores['hitrate']:.4f}"
        f" | MRR@{cfg.train.k}={test_scores['mrr']:.4f}"
        f" | NDCG@{cfg.train.k}={test_scores['ndcg']:.4f}"
    )
    wandb.log({f"test/{name}": value for name, value in test_scores.items()})
    wandb.finish()


if __name__ == "__main__":
    main()