## Jumbled Naive Bayes Classifier

This project demonstrates the operation of a modified Naive Bayes classifier, where the dimensionality of the space in which the original feature vectors are located is first reduced. The program splits the dataset into training and testing portions. For a number of dimensions, boxplots are constructed using random projection matrices. The program allows one to compare the performance of the new method (Jumbled Naive Bayes Classifier) ​​with the Chow-Liu, TAN, and classical Naive Bayes methods. In the case of Jumbled Naive Bayes, results are obtained for both the random matrix and the pre-trained one. Based on the obtained results, comparative graphs of accuracy (against the boxplots) and logarithmic time are constructed.

## Project Layout:
- Data/
- p1_make_train_test.py
- p2_unified_process_patients.py
- p3_chow_liu_patients_no_compression.py
- p4_naive_bayesian.py
- p5_matrix_selection.py
- p6_plots.py
- p7_time_plots.py
- requirements.txt

## Prerequisites:
- Visual Studio Code installed
- Git installed

## Installation:
```bash
cd <target_directory>
git clone https://github.com/Ivan-Denkov/Jumbled-NBC.git
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset:
Place dataset into root directory of the project

## Usage:
Run the files sequentially, starting with p1 and ending with p7

## Outputs:
Results (the following files) are written into root directoty.

- `sparse_vectors_patients_train.csv` (train data)
- `sparse_vectors_patients_test.csv` (test data)
- `accuracies_<current_datetime>_to_<smaller_dimensionality>_with_step_<step_for_boxplots>.csv` (boxplots data)
- `results_accuracy_time_<method>_small.csv` (name of method depends on the chosen one)