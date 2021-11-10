# -*- encoding: utf-8 -*-
import numpy as np

f = "../onehot_distribution.txt"
file = "../CBET_train.txt"
file2 = "../CBET_max.txt"
file3 = open("CBET-train.txt", "w")
file4 = open("CBET-label.txt", "w")
file5 = open("onehot_distribution.txt", "w")
data_size = 51240
shuffled_index = np.random.permutation(data_size)

all_3 = np.genfromtxt(f, dtype=str)
all_1 = []
with open(file) as f:
    lines = f.readlines()
    for line in lines:
        all_1.append(line)

all_2 = np.loadtxt(file2, dtype=str)


for i in range(data_size):
    file3.write(all_1[shuffled_index[i]])
    file4.write(all_2[shuffled_index[i]] + '\n')
    file5.write(' '.join(all_3[shuffled_index[i]]) + '\n')

file3.close()
file4.close()
file5.close()
