import polars as pl


events=pl.read_csv("dataset/raw/events.csv")

events=events.filter(pl.col("visitorid").is_in(
                     events.group_by('visitorid').len().filter(pl.col('len')>=5)['visitorid'])).sort('timestamp')


print(events.height,events['visitorid'].n_unique())