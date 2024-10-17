import requests
import time
import pandas as pd

RPC_URL = "http://localhost:8545"

def get_block(blk_num):
    hex_blk_num = hex(blk_num)
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_getBlockByNumber",
        "params": [hex_blk_num, True], 
        "id": 1
    }
    
    try:
        response = requests.post(RPC_URL, json=payload)
        response.raise_for_status() 
        block_data = response.json()

        if 'result' in block_data:
            return block_data['result']
        else:
            print("Block not found or error in response")
            return None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching block data: {e}")
        return None


def CalculateCumulativeAvgBurntFee(currentTotalBurntFee):
    start = time.time()
    data = pd.read_csv("../Data/block_data(Inc).csv")
    if data.shape[0] % 32 == 0:
        epoch = data.shape[0] // 32
        totalBurntFee = currentTotalBurntFee
        for i in range(df.shape[0]-32, data.shape[0]):
            totalBurntFee += data.iloc[i]['burntFee']
        avgBurntFee = totalBurntFee / data.shape[0]
        end = time.time()
        
        new_row = pd.DataFrame({
            "epoch": [epoch],
            "number": [int(data.iloc[-1]['number'])],
            "avgBurntFee": [avgBurntFee],
            "time": [end - start]
        })
        print(new_row)
        
        
        return new_row, totalBurntFee
    


try:
    df = pd.read_csv('../Data/block_data(Inc).csv')
except FileNotFoundError:
    df = pd.DataFrame(columns=['number', 'baseFeePerGas', 'gasUsed', 'burntFee'])
incCsvQuery = pd.DataFrame(columns=["epoch", "number", "avgBurntFee", "time"])

currentTotalBurntFee = 0
for blk_num in range(19000000, 19001024):
    block_info = get_block(blk_num)
    if block_info:
        new_row = pd.DataFrame({
            "number": [int(block_info["number"], 16)],
            "baseFeePerGas": [int(block_info["baseFeePerGas"], 16)],
            "gasUsed": [int(block_info["gasUsed"], 16)],
            "burntFee": [int(block_info["baseFeePerGas"], 16)*int(block_info["gasUsed"], 16)/1e18]
        })  
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv('../Data/block_data(Inc).csv', index=False)
       
        result = CalculateCumulativeAvgBurntFee(currentTotalBurntFee)
        if result:
            currentTotalBurntFee = result[1]
            incCsvQuery = pd.concat([incCsvQuery, result[0]], ignore_index=True)

incCsvQuery.to_csv("incCsvQuery.csv", index=False)