import pandas as pd
import math


def Shannon_entropy(proposer_count):
    if proposer_count == 0:
        return 0
    probabilities = [1/proposer_count] * proposer_count
    entropy = -sum(p * math.log2(p) for p in probabilities)
    return entropy


blk_data = pd.read_csv("../Data/blkDataFull.csv")
miner_data = blk_data.groupby("date")["miner"].nunique().reset_index()
miner_data["entropy"] = miner_data["miner"].apply(Shannon_entropy)
miner_data.to_csv(f"miner_entropy.csv", index=False)
