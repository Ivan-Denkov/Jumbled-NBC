#Applying Naive Bayes Classifier to data
import os
import pandas as pd
import csv
from sklearn.naive_bayes import BernoulliNB
import time
from sklearn.metrics import confusion_matrix
import numpy as np

#settings
volume = 3
step = 5
right_boundary = 200
train = f"sparse_vectors_patients_train_{volume}.csv"
test = f"sparse_vectors_patients_test_{volume}.csv"
result = f"results_accuracy_time_NB_small_{volume}.csv"
file_exists = os.path.isfile(result)

#preparing data
df_train = pd.read_csv(train, header=None)
dimensions = df_train.shape[1] - 1
X_train_full = np.ascontiguousarray(df_train.iloc[:, :dimensions].to_numpy(), dtype=np.float32)
y_train = df_train.iloc[:, dimensions].to_numpy(dtype=np.int8)

df_test = pd.read_csv(test, header=None)
X_test_full = np.ascontiguousarray(df_test.iloc[:, :dimensions].to_numpy(), dtype=np.float32)
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

            #classifier
            model = BernoulliNB(alpha=1.0)
            t0 = time.time()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
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