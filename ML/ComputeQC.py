import pandas as pd
import math

base_df = pd.read_csv("../Data/results(baseFee).csv")
base_df["uncertainty_log"] = base_df["epistemic_uncertainty"].apply(math.log)
base_df["quality"] = -1*base_df["uncertainty_log"]
base_df["cost(ms)"] = base_df["prediction_time"]*1000
base_df["quality:cost"] = base_df["quality"]/base_df["cost(ms)"]

epochs = base_df["epoch"].sample(n=5, random_state=123)
epochs = epochs.sort_values()

base_df_sample = base_df[base_df["epoch"]==epochs.iloc[0]]
for epoch in epochs[1:]:
    base_df_sample = pd.concat([base_df_sample, base_df[base_df["epoch"]==epochs.iloc[0]]])

df_pivot = base_df_sample.pivot(index='epoch', columns='model', values='quality:cost').reset_index()
df_pivot.to_csv("\QCBaseFee.csv", index=False)


priority_df = pd.read_csv("../Data/results(priorityFee).csv")
priority_df["uncertainty_log"] = priority_df["epistemic_uncertainty"].apply(math.log)
priority_df["quality"] = -1*priority_df["uncertainty_log"]
priority_df["cost(ms)"] = priority_df["prediction_time"]*1000
priority_df["quality:cost"] = priority_df["quality"]/priority_df["cost(ms)"]

epochs = priority_df["epoch"].sample(n=5, random_state=123)
epochs = epochs.sort_values()

priority_df_sample = priority_df[priority_df["epoch"]==epochs.iloc[0]]
for epoch in epochs[1:]:
    priority_df_sample = pd.concat([priority_df_sample, priority_df[priority_df["epoch"]==epochs.iloc[0]]])

df_pivot = priority_df_sample.pivot(index='epoch', columns='model', values='quality:cost').reset_index()
df_pivot.to_csv("\QCPriorityFee.csv", index=False)



