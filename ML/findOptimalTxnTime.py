import datetime
import pickle
import pandas as pd

def findOptimalTxnTime(features, model):
    iteration = 1
    optimal_time = None
    optimal_fee = float('inf') 

    while True:
        current_time = datetime.datetime.now()  
        candidate_times = [current_time + datetime.timedelta(minutes=30 * i) for i in range(2 ** iteration)]
        for candidate_time in candidate_times:
            predicted_fee = model.predict(features, candidate_time)
            if predicted_fee < optimal_fee:
                optimal_fee = predicted_fee
                optimal_time = candidate_time

        
        print("Optimal time to send transaction:", optimal_time, "with fee:", optimal_fee)
        
        
        if iteration >= 10 or (optimal_time and abs(optimal_fee - predicted_fee) < 0.001):
            break
        
        iteration += 1

    
with open("../Data/model.pkl", 'rb') as file:
    model = pickle.load(file)

eth_data = pd.read_csv("../Data/ethData.csv")
features = eth_data.tail(1)

value = int(input("Enter transacton value: "))
features['value'] = value

findOptimalTxnTime(features, model)
