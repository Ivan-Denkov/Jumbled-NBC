#Drawing log time plots for all methods
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#settings
volume = 3
max_ter = 4
MIN_DIV = 0.0000625

#Chow-Liu, TAN, Naive Bayesian and Jumbled Bayes (preparation)
cl = f"results_accuracy_time_Chow-Liu_small.csv"#_{volume}.csv"
tan = f"results_accuracy_time_TAN_small.csv"#_{volume}.csv"
nb = f"results_accuracy_time_NB_small.csv"#_{volume}.csv"
jum = f"results_accuracy_time_Jumbled_learned_matrix_small_limit_{max_ter}_iterations.csv"#_{volume}.csv"

cl_acc = pd.read_csv(cl).values
tan_acc = pd.read_csv(tan).values
nb_acc = pd.read_csv(nb).values
jum_acc = pd.read_csv(jum).values

plt.figure(figsize=(10, 6))
plt.xlabel("Розмірність")

plt.yscale("log")
plt.ylabel("Час, хвилини (логарифмічна шкала)")

#Chow-Liu, TAN, Naive Bayesian and Jumbled Bayes (plots)
plt.plot(cl_acc[:, 0], cl_acc[:, 2], marker="o", color="blue", label="Chow-Liu")
plt.plot(tan_acc[:, 0], tan_acc[:, 2], marker="s", color="red", label="TAN")
nb_acc[:, 2] = np.where(nb_acc[:, 2] > 0, nb_acc[:, 2], MIN_DIV)
plt.plot(nb_acc[:, 0], nb_acc[:, 2], marker="^", color="green", label="NBC")
plt.plot(jum_acc[:, 0], jum_acc[:, 2], marker="D", color="black", label="Jumbled Bayes")

xticks = cl_acc[::2, 0].astype(int)
plt.xticks(xticks, [str(x) for x in xticks], rotation=45)
plt.title("Порівняльний графік для часу роботи Jumbled Bayes і класичних методів")

plt.legend()
plt.grid(True, axis="y", alpha=0.3)
plt.show()