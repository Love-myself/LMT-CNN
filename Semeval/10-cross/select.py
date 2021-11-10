# -*- encoding: utf-8 -*-
import numpy as np

data_train = "../Shuffle/semeval_train1.txt"
data_label = "../Shuffle/semeval_label_distribution.txt"

data_size = 1250
split_size = int(data_size*0.1)

All_train = []
with open(data_train, encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        All_train.append(line)
All_train = np.array(All_train, dtype=str)
All_label = np.loadtxt(data_label, delimiter=' ')


for i in range(10):
    # cross_test_data = All_train[i*split_size:(i+1)*split_size]
    # cross_test_label = All_label[i*split_size:(i+1)*split_size]
    # cross_train_data = np.concatenate((All_train[:i*split_size], All_train[(i+1)*split_size:]), axis=0)
    # cross_train_label = np.concatenate((All_label[:i*split_size], All_label[(i+1)*split_size:]), axis=0)
    cross_train_data = All_train
    cross_train_label = All_label
    np.save("train_data.npy", cross_train_data)
    np.save("train_label.npy", cross_train_label)
    # np.save("test_data.npy", cross_test_data)
    # np.save("test_label.npy", cross_test_label)

# a = np.load("cross0/train_data.npy")
# a = a.tolist()
# for line in a[0:1]:
#     print(line)
# b = np.load("cross0/train_label.npy")
# c = np.load("cross0/test_data.npy")
# d = np.load("cross0/test_label.npy")
# print(a.shape, b.shape, c.shape, d.shape)
# print(a)
# print(b)
# print(c)
# print(d)
