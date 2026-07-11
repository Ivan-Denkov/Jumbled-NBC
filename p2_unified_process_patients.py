#Jumbled Naive Bayes Booster: gathering boxplots statististics and import in p5_matrix_selection.py
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import csv
from datetime import datetime

#probabilities of desease for 0 and 1
def probs_preparation(vectors, labels):
    y = (labels == 2).astype(np.uint8)
    cnt1_total = y.sum()
    cnt1_in_1 = (y[:, None] * vectors).sum(axis=0) 
    cnt_in_1 = vectors.sum(axis=0)
    cnt1_in_0 = cnt1_total - cnt1_in_1
    cnt_in_0 = vectors.shape[0] - cnt_in_1
    p2_if_0 = cnt1_in_0 / np.maximum(cnt_in_0, 1)
    p2_if_1 = cnt1_in_1 / np.maximum(cnt_in_1, 1)
    return [p2_if_0, p2_if_1]

#predicting desease/non-desease
def prediction(probabilities, eps, X):
    p0 = np.clip(probabilities[0], eps, 1 - eps)
    p1 = np.clip(probabilities[1], eps, 1 - eps)
    logit_0 = np.log(p0) - np.log(1 - p0)
    logit_1 = np.log(p1) - np.log(1 - p1)
    scores = logit_0 + X * (logit_1 - logit_0) #(1 - X) * logit_0 + X * logit_1
    scores = scores.sum(axis=1)
    predicted = np.where(scores >= 0, 2, 1)
    return predicted

if __name__ == "__main__":
    #preparing materials
    eps = 1e-9
    volume = 3
    step = 5
    right_boundary = 200
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    file_out = f"accuracies_{timestamp}_to_{right_boundary}_with_step_{step}.csv"

    df_learn = pd.read_csv(f"sparse_vectors_patients_train_{volume}.csv", header=None)
    dimensions = df_learn.shape[1] - 1
    X = df_learn.iloc[:, :dimensions].values
    X = np.ascontiguousarray(X, dtype=np.float32)
    labels = df_learn.iloc[:, dimensions].values
    labels = labels.astype(np.int8, copy=False)

    df_test = pd.read_csv(f"sparse_vectors_patients_test_{volume}.csv", header=None)
    X_test = df_test.iloc[:, :dimensions].values
    X_test = np.ascontiguousarray(X_test, dtype=np.float32)
    labels_test = df_test.iloc[:, dimensions].values
    labels_test = labels_test.astype(np.int8, copy=False)

    #process
    for second_dim in range(step, right_boundary + step, step):
        accuracies = []
        print("Dimension: ", second_dim)

        for number in range(1, 501):
            #print("Iteration: ", number)
            #matrix generation
            M = np.random.choice([-1, 1], size=(dimensions, second_dim))
            M = np.ascontiguousarray(M, dtype=np.float32)

            #model fit
            X_bin = (X @ M >= 0).astype(np.uint8)
            probabilities = probs_preparation(X_bin, labels)

            #model predict
            X_bin_test = (X_test @ M >= 0).astype(np.uint8)
            predicted = prediction(probabilities, eps, X_bin_test)

            #calculating precision
            cm = confusion_matrix(labels_test, predicted, labels=[1, 2])
            cm_percent = cm / cm.sum(axis=1, keepdims=True) * 100
            accuracies.append(np.diag(cm_percent).mean())
        
        with open(file_out, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(accuracies)

    print("Complete!!!")