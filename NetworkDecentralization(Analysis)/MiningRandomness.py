import pandas as pd
import numpy as np

def ShannonEntropy(data):
    shannon = 0
    for p in data:
        if p == 0:
            continue
        else:
            shannon += p * np.log2(p)
    return -shannon

def getDistribution(data):
    blocks = data['Blocks mined/proposed']
    sum = blocks.sum()
    distribution = blocks / sum
    return distribution


blk_data = pd.read_csv("../Data/blkDataFull.csv")
blk_data['date'] = pd.to_datetime(blk_data['date'])
blk_data['Week'] = blk_data['date'].dt.to_period('W')  
blocks_week = blk_data.groupby(['Week', 'miner']).size().reset_index(name='Blocks mined/proposed')

blocks_week.columns = ['Week', 'Miner', 'Blocks mined/proposed']

mining_randomness = pd.DataFrame(columns=['Week', 'Entropy'])

for week in blocks_week['Week'].unique():
    week_data = blocks_week[blocks_week['Week'] == week]
    entropy = ShannonEntropy(getDistribution(week_data))
    new_row = pd.DataFrame({'Week': [week], 'Entropy': [entropy]})
    mining_randomness = pd.concat([mining_randomness, new_row])

mining_randomness.to_csv(f"mining_randomness.csv", index=False)
