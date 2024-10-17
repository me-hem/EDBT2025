import requests
import time
import pandas as pd
import psycopg2


# Set up connection to the local Geth node
RPC_URL = "http://localhost:8545"

# Function to get block data from Geth
def GetBlock(blk_num):
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


def IngestData(block, df):
    number = int(block['number'], 16)  
    base_fee = int(block.get('baseFeePerGas', '0x0'), 16)/ 1e18  
    gas_used = int(block['gasUsed'], 16)
    burnt_fee = (base_fee * gas_used)   

    conn = psycopg2.connect(
        host="localhost",
        database="EthData",
        user="postgres",
        password="cd  "
    )
    cur = conn.cursor()

    query = """
    INSERT INTO BlockData (number, baseFeePerGas, gasUsed, burntFee)
    VALUES (%s, %s, %s, %s)
    """
    values = (number, base_fee, gas_used, burnt_fee)

    try:
        cur.execute(query, values)
        conn.commit()
        start = time.time()
        cur.execute("SELECT * FROM CalculateCumulativeAvgBurntFee();")
        cumulative_result = cur.fetchone()
        end = time.time()
        if cumulative_result:
            cur.execute("SELECT pg_database_size(%s);", ("EthData",))
            db_size = cur.fetchone()[0]  
            db_size_mb = db_size / (1024 * 1024)
            
            latest_epoch, avg_burnt_fee = cumulative_result
            print(f"Latest Epoch: {latest_epoch}, Average Burnt Fee: {avg_burnt_fee}")
            new_row = pd.DataFrame({
                "epoch": [latest_epoch],
                "number": [number],
                "avgBurntFee": [avg_burnt_fee],
                "time": [end - start],
                "size": [db_size_mb]
            })
            df = pd.concat([df, new_row], ignore_index=True)
    except psycopg2.Error as e:
        print(f"Error inserting block {number}: {e}")
    finally:
        cur.close()
        conn.close()
        
    return df


fullViewQuery = pd.DataFrame(columns=["epoch", "number", "avgBurntFee", "time", "size"])
# Can be modified to fetch latest blocks
for blk_num in range(19000000, 19001024):
    block = GetBlock(blk_num)
    if block:
        fullViewQuery = IngestData(block, fullViewQuery)
fullViewQuery.to_csv("fullViewQuery.csv", index=False)