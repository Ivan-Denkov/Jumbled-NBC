#Creating train and test datasets
import pandas as pd

TRAIN_SIZE = 7200
volume = 3
df = pd.read_csv("sparse_vectors_patients.csv", header=None)

label_col = df.columns[-1]
df_1 = df[df[label_col] == 1]
df_2 = df[df[label_col] == 2]

n_per_class = TRAIN_SIZE // 2

train_1 = df_1.sample(n=n_per_class)
train_2 = df_2.sample(n=n_per_class)

train_df = pd.concat([train_1, train_2])
train_df = train_df.sample(frac=1)
test_df = df.drop(train_df.index)

train_df.to_csv(f"sparse_vectors_patients_train_{volume}.csv", index=False, header=False)
test_df.to_csv(f"sparse_vectors_patients_test_{volume}.csv", index=False, header=False)