# -*- encoding: utf-8 -*-
import pandas as pd
import numpy as np
import xlrd

NRC_file = "NRC.xls"
Emo_file = "emosenticnet.xls"

table1 = xlrd.open_workbook(NRC_file).sheets()[0]
row1 = table1.nrows
# print(table1.row_values(3))
NRC_index = []
NRC_book = []
for i in range(row1):
    a = table1.row_values(i)
    if a[0] == 0:
        a[0] = "false"
    elif a[0] == 1:
        a[0] = "true"
    NRC_index.append(a[0])
    NRC_book.append(a[1:])
NRC_index.remove(NRC_index[0])
NRC_book.remove(NRC_book[0])

# Emo_file
table2 = xlrd.open_workbook(Emo_file).sheets()[0]
row2 = table2.nrows
Emo_index = []
Emo_book = []
for i in range(row2):
    a = table2.row_values(i)
    if a[0] == 0:
        a[0] = "false"
    elif a[0] == 1:
        a[0] = "true"
    Emo_index.append(a[0])
    Emo_book.append(a[1:])
Emo_index.remove(Emo_index[0])
Emo_book.remove(Emo_book[0])

# # print(Emo_index)
# # print(Emo_book)
# # print(NRC_index)
# # print(NRC_book)

# 合并
word_index = []
word_dict = []
cnt = 0
vis = []
for word in NRC_index:
    if word in Emo_index:
        pos = Emo_index.index(word)
        lt = []
        for i in range(8):
            if NRC_book[cnt][i] == Emo_book[pos][i]:
                lt.append(NRC_book[cnt][i])
            else:
                lt.append(1.0)
        lt.append(NRC_book[cnt][6])
        word_dict.append(lt)
        vis.append(pos)
    else:
        word_dict.append(NRC_book[cnt])
    word_index.append(word)
    cnt += 1


all = range(len(Emo_index))
need_add = [w for w in all if w not in vis]

for i in need_add:
    word_index.append(Emo_index[i])
    Emo_book[i].append(0)
    word_dict.append(Emo_book[i])

# print(len(vis))
# print(len(need_add))
print(len(word_index))
print(word_index)
print(word_dict)

word_index = np.array(word_index, dtype=str)
word_dict = np.array(word_dict, dtype=int)

# print(word_dict.shape)
np.save("word_index.npy", word_index)
np.save("word_dict.npy", word_dict)

