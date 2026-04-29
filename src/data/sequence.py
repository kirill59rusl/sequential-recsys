import polars as pl

MAX_LEN=50

def build_sequences(lf:pl.LazyFrame):
    
    lfc=(lf.sort(["user_id","timestamp"])
             .group_by("user_id").agg([
                 pl.col("event_id").alias("event_sequence"),
                 pl.col("item_id").alias("item_sequence"),
                 pl.col("session").alias("session_sequence"),
                 pl.col("new_session").alias("new_sequence"),
                 pl.col("delta_minute").alias("delta_sequence"),
                 pl.col("delta_bucket").alias("bucket_sequence")
             ]).with_columns(
                 pl.col('event_sequence').list.len().alias("sequence_len")
             )
    )

    return lfc



def main():
    dataset=pl.scan_parquet("dataset/processed/full_data.parquet")
    dataset=build_sequences(dataset).collect()
    dataset.write_parquet("dataset/processed/sequences.parquet")

if __name__=='main':
    main()


