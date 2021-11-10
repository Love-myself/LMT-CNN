# -*- encoding: utf-8 -*-
import numpy as np

f = "../new_semeval_label_distribution.txt"
file = "../semeval_train1.txt"
file3 = open("semeval_train1.txt", "w")
file5 = open("semeval_label_distribution.txt", "w")
data_size = 1250
shuffled_index = np.random.permutation(data_size)     #生成随机序列

all_3 = np.genfromtxt(f, dtype=str)  #从文件读取的字符串序列要转换为其他类型数据时需设置dtype参数 genfromtxt函数创建数组表格数据
all_1 = []
with open(file) as f:
    lines = f.readlines()
    for line in lines:
        all_1.append(line)

#all_2 = np.loadtxt(file2, dtype=str) #用于从文本加载数据。文本文件中的每一行必须含有相同的数据


for i in range(data_size):   #打乱数据集的顺序
    file3.write(all_1[shuffled_index[i]])
    #file4.write(all_2[shuffled_index[i]] + '\n')
    file5.write(' '.join(all_3[shuffled_index[i]]) + '\n')

file3.close()
#file4.close()
file5.close()
