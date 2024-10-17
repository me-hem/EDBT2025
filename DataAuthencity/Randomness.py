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

# Full node data   
shannonFull = pd.read_csv("../Data/shannonFullEthSample.csv")
date = shannonFull.loc[0, 'date']
txnValues = [pd.to_numeric(shannonFull.loc[0, 'nor_txnValue'], errors='coerce')]  

dailyShannonData = pd.DataFrame(columns=['date', 'shannon_entropy'])

for txn in range(1, shannonFull.shape[0]):
    current_date = shannonFull.loc[txn, 'date']
    
    if date != current_date:        
        shannon_entropy = ShannonEntropy(txnValues)       
        new_row = pd.DataFrame({
            'date': [date],
            'shannon_entropy': [shannon_entropy]
        })
        dailyShannonData = pd.concat([dailyShannonData, new_row], ignore_index=True)
        date = current_date
        txnValues = [pd.to_numeric(shannonFull.loc[txn, 'nor_txnValue'], errors='coerce')]
    else:
        txnValues.append(pd.to_numeric(shannonFull.loc[txn, 'nor_txnValue'], errors='coerce'))

shannon_entropy = ShannonEntropy(txnValues)
new_row = pd.DataFrame({
    'date': [date],
    'shannon_entropy': [shannon_entropy]
})
dailyShannonData = pd.concat([dailyShannonData, new_row], ignore_index=True)
dailyShannonData.to_csv("/dailyFullEthSample_entropy.csv", index=False)



# Api data
shannonApi = pd.read_csv("../Data/shannonApiEthSample.csv")
date = shannonApi.loc[0, 'date']
txnValues = [pd.to_numeric(shannonApi.loc[0, 'nor_txnValue'], errors='coerce')]

dailyShannonData = pd.DataFrame(columns=['date', 'shannon_entropy'])

for txn in range(1, shannonApi.shape[0]):
    current_date = shannonApi.loc[txn, 'date']
    
    if date != current_date:        
        shannon_entropy = ShannonEntropy(txnValues)       
        new_row = pd.DataFrame({
            'date': [date],
            'shannon_entropy': [shannon_entropy]
        })
        dailyShannonData = pd.concat([dailyShannonData, new_row], ignore_index=True)
        date = current_date
        txnValues = [pd.to_numeric(shannonApi.loc[txn, 'nor_txnValue'], errors='coerce')]
    else:
        txnValues.append(pd.to_numeric(shannonApi.loc[txn, 'nor_txnValue'], errors='coerce'))

shannon_entropy = ShannonEntropy(txnValues)
new_row = pd.DataFrame({
    'date': [date],
    'shannon_entropy': [shannon_entropy]
})
dailyShannonData = pd.concat([dailyShannonData, new_row], ignore_index=True)
dailyShannonData.to_csv("dailyApiEthSample_entropy.csv", index=False)

