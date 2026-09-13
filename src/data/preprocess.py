from pathlib import Path
import polars as pl

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf


def sessionize(lf:pl.LazyFrame,gap:int=30):
    #добавление сессий и временных промежутков между действиями
    dfl=lf
    dfc=(dfl.sort(["visitorid", "timestamp"])
        .with_columns(((pl.col("timestamp")-pl.col("timestamp").shift(1))/1000/60)
                               .over("visitorid").cast(pl.Float32).alias("gap"))
        .with_columns((pl.when((pl.col("gap").is_not_null()) & (pl.col("gap")<=gap))
                                 .then(0)
                                 .otherwise(1)).alias("new_session"))

        .with_columns((pl.col("new_session").cum_sum()-1).over("visitorid").alias("session"))
    )
    return dfc.with_columns(pl.col("gap").fill_null(0).cast(pl.Float32))


def k_core_filter(lf: pl.LazyFrame ,k: int=5,verbose:bool=False):
    prev=None
    itera=0
    while True:
        itera+=1
        df=(lf
            .join(lf.group_by("visitorid")
                  .len()
                  .filter(pl.col("len")>=k),on='visitorid',how='semi')
            .join(lf.group_by("itemid")
                  .len()
                  .filter(pl.col("len")>=k),on='itemid',how='semi').collect())
        curr = (
            df.height,
            df["visitorid"].n_unique(),
            df["itemid"].n_unique(),
        )
        if verbose==True:
            print(
                f"{itera}: "
                f"rows={curr[0]} "
                f"users={df['visitorid'].n_unique()} "
                f"items={df['itemid'].n_unique()}"
            )
        if curr==prev:
            break
        lf=df.lazy()
        prev=curr
    return df.lazy()


def encode_ids(lf:pl.LazyFrame,col='visitorid',name='user_id',artifact_dir: Path = None):
    mapping=(lf.select(col)
        .unique()
        .sort(col)
        .with_row_index(name,offset=1)
    )
    #сохраняем кодировку в parquet file
    to_save = mapping.collect()

    artifact_dir = artifact_dir or Path("dataset/artifact")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    to_save.write_parquet(artifact_dir / f"{name}.parquet")
    return lf.join(mapping, on=col)

def encode_gap(lf:pl.LazyFrame):
    
    dfc=lf.with_columns(pl
                        .when(pl.col("new_session")==1).then(0) 
                        .when(pl.col("gap")<=1).then(1) 
                        .when(pl.col("gap")<=5).then(2) 
                        .when(pl.col("gap")<=30).then(3) 
                        .when(pl.col("gap")<=2*60).then(4) 
                        .when(pl.col("gap")<=24*60).then(5) 
                        .when(pl.col("gap")<=7*24*60).then(6) 
                        .otherwise(7).alias('delta_bucket'),
                        pl.col("gap").alias("delta_minute")).drop("gap")
    return dfc

EVENT_MAP={
    'view':0,
    'addtocart':1,
    'transaction':2
}

def encode_action(lf:pl.LazyFrame,mapping=EVENT_MAP):
    
    dfc=lf.with_columns(pl.col("event")
                        .replace(mapping)
                        .cast(pl.Int8)
                        .alias('event_id')).drop("event")
    
    return dfc


def build_dataset(
    lf: pl.LazyFrame,
    k_core: int = 3,
    session_gap: int = 30,
):
    lf = k_core_filter(lf, k=k_core).drop("transactionid")
    lf = sessionize(lf, gap=session_gap)
    lf = encode_ids(lf).drop("visitorid")
    lf = encode_ids(lf, "itemid", "item_id").drop("itemid")
    lf = encode_gap(lf)
    lf = encode_action(lf)
    return lf.collect()

@hydra.main(
    version_base=None,
    config_path="../../conf", 
    config_name="config",
)
def main(cfg: DictConfig):
    root = Path(get_original_cwd())

    events = pl.scan_csv(root / "dataset/raw/events.csv")

    events = build_dataset(
        events,
        k_core=cfg.data.k_core,
        session_gap=cfg.data.session_gap,
    )

    events.write_parquet(root / "dataset/processed/full_data.parquet")

if __name__=='__main__':
    main()