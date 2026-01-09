import pandas as pd

# 1. Load the full training set
df = pd.read_csv("train.csv")

# 2. Separate Consistent (1) and Contradict (0)
# Note: Check the exact column name for labels. Usually 'label' or 'target'.
# Assuming 'label' based on typical datasets. If it fails, print(df.columns) to check.
consistent = df[df['label'] == 'consistent'].head(5)
contradict = df[df['label'] == 'contradict'].head(5)

# 3. Combine them
mini_train = pd.concat([consistent, contradict])

# 4. Save to new CSV
mini_train.to_csv("mini_train.csv", index=False)

print(f"Created mini_train.csv with {len(mini_train)} rows.")
print(mini_train[['id', 'label']]) # Verify you have mixed labels