#NBC with matrices "learning"
import pandas as pd
import numpy as np
from p2_unified_process_patients import probs_preparation, prediction
from sklearn.metrics import confusion_matrix
import csv
import os
import time

#Config
volume = 3
train = f"sparse_vectors_patients_train_{volume}.csv"
test = f"sparse_vectors_patients_test_{volume}.csv"
max_iter = 4
result = f"results_accuracy_time_Jumbled_learned_matrix_small_limit_{max_iter}_iterations_{volume}.csv"
file_exists = os.path.isfile(result)
step = 5
right_boundary = 200
MIN_DIV = 0.0000625
eps = 1e-9

#Function to get accuracy
def get_accuracy(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xte: np.ndarray,
    yte: np.ndarray,
    M: np.ndarray,
    eps: float = 1e-9
) -> float:
    X_bin_train = (Xtr @ M >= 0).astype(np.uint8)
    probs = probs_preparation(X_bin_train, ytr)
    X_bin_test = (Xte @ M >= 0).astype(np.uint8)
    y_pred = prediction(probs, eps, X_bin_test)
    
    cm = confusion_matrix(yte, y_pred, labels=[1, 2])
    cm_percent = cm / cm.sum(axis=1, keepdims=True) * 100
    
    return np.diag(cm_percent).mean()

#Data and file preparation
with open(result, "a", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(["Dimension", "Accuracy", "Time", "Initial_Accuracy"])

df_train = pd.read_csv(train, header=None)
dimensions = df_train.shape[1] - 1
X_train = np.ascontiguousarray(df_train.iloc[:, :dimensions].to_numpy(), dtype=np.float32)
y_train = df_train.iloc[:, dimensions].to_numpy().astype(np.int8)

df_test = pd.read_csv(test, header=None)
X_test = np.ascontiguousarray(df_test.iloc[:, :dimensions].to_numpy(), dtype=np.float32)
y_test = df_test.iloc[:, dimensions].to_numpy().astype(np.int8)


for second_dim in range(step, right_boundary + step, step):
    print("Second dimension: ", second_dim)

    try:
        #Initial matrix
        A = np.random.choice([-1, 1], size=(dimensions, second_dim)).astype(np.float32)

        #Test matrix from the scratch
        acc_before = get_accuracy(X_train, y_train, X_test, y_test, A, eps)

        #cycle for one second dimension
        coincide = False
        counter = 0

        t0 = time.time()
        Z_A = (X_train @ A >= 0).astype(np.uint8)

        while not coincide and counter < max_iter:
            counter += 1
            print("Iteration: ", counter)
            
            B = np.random.choice([-1, 1], size=(dimensions, second_dim)).astype(np.float32)
            Z_B = (X_train @ B >= 0).astype(np.uint8)
            Z = np.hstack([Z_A, Z_B])

            patients_code1 = Z[y_train == 1]
            patients_code2 = Z[y_train == 2]

            percent1 = patients_code1.mean(axis=0) * 100
            percent2 = patients_code2.mean(axis=0) * 100

            r1 = percent2 / (percent1 + MIN_DIV)
            r2 = (100 - percent2) / (100 - percent1 + MIN_DIV)

            importance = np.abs(r1 - r2)
            order = np.argsort(-importance)
            
            '''for new_pos, old_idx in enumerate(order):
                print(f"{new_pos} position from {old_idx}")'''

            C = np.hstack([A, B])
            C_sorted = C[:, order]
            Z_sorted = Z[:, order]

            A_new = C_sorted[:, :second_dim]
            Z_A_new = Z_sorted[:, :second_dim]

            if np.array_equal(A, A_new):
                coincide = True
            else:
                A = A_new.copy()
                Z_A = Z_A_new.copy()

        #Test "learned" matrix
        acc_after = get_accuracy(X_train, y_train, X_test, y_test, A, eps)
        t1 = time.time()
        time_min = (t1 - t0) / 60

        #CSV output
        with open(result, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([second_dim, acc_after, time_min, acc_before])
            f.flush()
    
    except Exception as e:
        print(f"Ошибка на размерности {second_dim}: {type(e).__name__}: {e}")
        break

print("Done!")