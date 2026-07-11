#Applying Chow-Liu/TAN methods to data
import pandas as pd
import pyagrum.skbn as skbn
import numpy as np
from sklearn.metrics import confusion_matrix
import time
import csv
import os

#settings
volume = 3
step = 5
right_boundary = 200
train = f"sparse_vectors_patients_train_{volume}.csv"
test = f"sparse_vectors_patients_test_{volume}.csv"
method = "TAN" #"Chow-Liu"
result = f"results_accuracy_time_{method}_small_{volume}.csv"
file_exists = os.path.isfile(result)

#preparing data
df_train = pd.read_csv(train, header=None)
dimensions = df_train.shape[1] - 1
X_train_full = np.ascontiguousarray(df_train.iloc[:, :dimensions].to_numpy(), dtype=np.int8)
y_train = df_train.iloc[:, dimensions].to_numpy(dtype=np.int8)

df_test = pd.read_csv(test, header=None)
X_test_full = np.ascontiguousarray(df_test.iloc[:, :dimensions].to_numpy(), dtype=np.int8)
y_test = df_test.iloc[:, dimensions].to_numpy(dtype=np.int8)

with open(result, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["Dimension", "Accuracy", "Time"])

    for dim in range(step, right_boundary + step, step):
        print("Dimension: ", dim)
        try:
            #taking first "dim" columns
            X_train = X_train_full[:, :dim]
            X_test = X_test_full[:, :dim]

            #excluding "only zero" columns
            non_const_mask = X_train.std(axis=0) > 0

            X_train = X_train[:, non_const_mask]
            X_test = X_test[:, non_const_mask]

            print(f"Used features: {X_train.shape[1]} / {dim}")

            #classifier
            clf = skbn.BNClassifier(
                learningMethod=method,
                prior="Smoothing",
                priorWeight=1.0,
                usePR=False
            )
            t0 = time.time()
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            t1 = time.time()
            time_min = (t1 - t0) / 60

            #statistics
            cm = confusion_matrix(y_test, y_pred, labels=[1, 2])
            cm_percent = cm / cm.sum(axis=1, keepdims=True) * 100
            total_acc = np.diag(cm_percent).mean()
            writer.writerow([dim, total_acc, time_min])
            f.flush()

            print("Done")
        
        except Exception as e:
            f.flush()
            print(f"Ошибка на размерности {dim}: {type(e).__name__}: {e}")
            break

print("Complete!!!")