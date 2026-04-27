import polars as pl
from torch.utils.data import Dataset,DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence

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


class SequenceDataset(Dataset):

    def __init__(self, df, max_len=50): #надо заменить
        self.samples=[]
        self.max_len=max_len
        self.rows=df.to_dicts()

        for user_idx,row in self.rows:
            seq_len=row['sequence_len']

            for pos in range(1,seq_len):
                user_id=row['user_id']
                self.samples.append((user_idx,pos))


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        user_idx,pos=self.samples[idx]
        left=max(0,pos-self.max_len)
        row=self.rows[user_idx]


        events=torch.tensor(row['event_sequence'][left:pos],dtype=torch.long)
        items=torch.tensor(row['item_sequence'][left:pos],dtype=torch.long)
        sessions=torch.tensor(row['session_sequence'][left:pos],dtype=torch.long)
        delta=torch.tensor(row['delta_sequence'][left:pos],dtype=torch.float32)
        bucket=torch.tensor(row['bucket_sequence'][left:pos],dtype=torch.long)
        target=torch.tensor(row['item_sequence'][pos],dtype=torch.long)
        return {
            'item_seq': items,
            'events_seq': events,
            'session_seq': sessions,
            'delta_seq': delta,
            'bucket_seq': bucket,
            'target': target,
        }
    
def collate_fn(batch): #функция для стыковки тензоров разных размеров при помощи паддинга
    
    keys = [
        'item_seq',
        'events_seq',
        'session_seq',
        'delta_seq',
        'bucket_seq'
    ]

    out = {}

    lengths = torch.tensor(
        [len(x['item_seq']) for x in batch],
        dtype=torch.long
    )

    for key in keys: #проходимся по ключам -> далее берём все последовательности и соединяем их при помощи pad_sequence
        seqs = [x[key] for x in batch]

        out[key] = pad_sequence( #вот тут
            seqs,
            batch_first=True,
            padding_value=0
        )

    out['target_seq'] = torch.stack( #соединяем все таргеты - проблем нет
        [x['target_seq'] for x in batch]
    )

    out['lengths'] = lengths

    mask = torch.arange(lengths.max().item())[None, :] < lengths[:, None]#маска для padding чтобы знал где настоящая
    out['mask'] = mask

    return out





dataset=pl.scan_parquet("dataset/processed/full_data.parquet")
dataset=build_sequences(dataset).collect()
#dataset.write_parquet("dataset/processed/sequences.parquet")
dataset=SequenceDataset(dataset)
loader = DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    collate_fn=collate_fn
)



