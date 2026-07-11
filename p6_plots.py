#Drawing accuracy plots for all methods with Jumbled Bayes boxplots as background
import pandas as pd
import matplotlib.pyplot as plt

#settings
acc = []
step = 5
right_boundary = 200
max_ter = 4

input = "accuracies_20260401_1450_to_200_with_step_5.csv"
accuracies = pd.read_csv(input, header=None).values

for accuracy in accuracies:
    acc.append(accuracy)

dimensions_small = list(range(step, right_boundary + step, step))

#Chow-Liu, TAN, Naive Bayesian and Jumbled Bayes (preparation)
cl = f"results_accuracy_time_Chow-Liu_small.csv"
tan = f"results_accuracy_time_TAN_small.csv"
nb = f"results_accuracy_time_NB_small.csv"
jum = f"results_accuracy_time_Jumbled_learned_matrix_small_limit_{max_ter}_iterations.csv"

cl_acc = pd.read_csv(cl).values
tan_acc = pd.read_csv(tan).values
nb_acc = pd.read_csv(nb).values
jum_acc = pd.read_csv(jum).values

plt.figure(figsize=(15, 8))
plt.xlabel("Розмірність")
plt.ylabel("Точність")

#Jumbled Bayes boxplots
plt.boxplot(acc, positions=dimensions_small, widths=4, showfliers=True, medianprops={"color": "magenta"})

#Chow-Liu, TAN, Naive Bayesian and Jumbled Bayes (plots)
plt.plot(cl_acc[:, 0], cl_acc[:, 1], marker="o", color="blue", label="Chow-Liu")
plt.plot(tan_acc[:, 0], tan_acc[:, 1], marker="s", color="red", label="TAN")
plt.plot(nb_acc[:, 0], nb_acc[:, 1], marker="^", color="green", label="NBC")
plt.plot(jum_acc[:, 0], jum_acc[:, 3], marker="v", color="orange", label="Jumbled Bayes (початкова матриця)")
plt.plot(jum_acc[:, 0], jum_acc[:, 1], marker="*", color="black", label="Jumbled Bayes ('навчена' матриця)")

xticks = dimensions_small[::2]
plt.xticks(xticks, [str(x) for x in xticks], rotation=45)
plt.title(f"Порівняльний графік для Jumbled Bayes і класичних методів")

plt.legend()
plt.grid(True, axis="y", alpha=0.3)
plt.show()