import pandas as pd

blk_data = pd.read_csv(f"../Data/blkDataFull.csv", index=False)

miner_data = blk_data.groupby("date")["miner"].nunique().reset_index()
miner_data.to_csv(f"miner_distribution.csv", index=False)
