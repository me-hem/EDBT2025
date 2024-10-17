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


def CalculateCumulativeAvgBurntFee():
    start = time.time()
    data = pd.read_csv("../Data/block_data(Full).csv")
    if data.shape[0] % 32 == 0:
        epoch = data.shape[0] // 32
        avgBurntFee = 0
        start = time.time()
        for i in range(data.shape[0]):
            avgBurntFee += data.iloc[i]['burntFee']
        avgBurntFee /= data.shape[0]
        end = time.time()
        
        new_row = pd.DataFrame({
            "epoch": [epoch],
            "number": [int(data.iloc[-1]['number'])],
            "avgBurntFee": [avgBurntFee],
            "time": [end - start]
        })
        print(new_row)
        
        return new_row


try:
    df = pd.read_csv('../Data/block_data(Full).csv')
except FileNotFoundError:
    df = pd.DataFrame(columns=['number', 'baseFeePerGas', 'gasUsed', 'burntFee'])
fullCsvQuery = pd.DataFrame(columns=["epoch", "number", "avgBurntFee", "time"])

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
        df.to_csv('../Data/block_data(Full).csv', index=False)
       
        fullCsvQuery = pd.concat([fullCsvQuery, CalculateCumulativeAvgBurntFee()], ignore_index=True)
fullCsvQuery.to_csv("fullCsvQuery.csv", index=False)