import pandas as pd
import numpy as np

def calculate_euclidean_distance(fullNodeDaily, apiDaily):
    merged_df = pd.merge(fullNodeDaily, apiDaily, on='date', suffixes=('_fullNode', '_api'))
    
    merged_df['euclidean_distance'] = np.sqrt(
        (merged_df['avg_gasUsed_fullNode'] - merged_df['avg_gasUsed_api'])**2 +
        (merged_df['avg_gasPrice_fullNode'] - merged_df['avg_gasPrice_api'])**2 +
        (merged_df['avg_txnValue_fullNode'] - merged_df['avg_txnValue_api'])**2
    )
    
    new_df = merged_df[['date', 'euclidean_distance']]
    
    return new_df


fullNodeDaily = pd.read_csv("../Data/dailyFullEthSample.csv")
apiDaily = pd.read_csv("../Data/dailyApiEthSample.csv")

distances_df = calculate_euclidean_distance(fullNodeDaily, apiDaily)

distances_df.to_csv("euclideanDistanceEth.csv", index=False)
