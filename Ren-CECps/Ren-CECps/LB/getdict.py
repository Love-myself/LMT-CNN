# -*- encoding: utf-8 -*-
import pandas as pd
import numpy as np
import xlrd

NRC_file = "NRC.xls"
Emo_file = "emosenticnet.xls"

table1 = xlrd.open_workbook(NRC_file).sheets()[0]
row1 = table1.nrows
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
print(len(NRC_index))

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
print(len(Emo_book))

word_index = []
word_dict = []
cnt=0

for word in NRC_index:
    x=NRC_index.index(word)
    if word in Emo_index:
        y = Emo_index.index(word)
        #print(cnt,x,y,word)
        lt=[]
        for i in range(6):
            if NRC_book[x][i] == Emo_book[y][i]:
               lt.append(NRC_book[x][i])
            else:
                lt.append(1.0)
    else:
        lt=NRC_book[x][:6]
    if 1.0 in lt:
        word_dict.append(lt)
        word_index.append(word)

for word in Emo_index:
    if word not in NRC_index:
        x=Emo_index.index(word)
        if 1.0 in Emo_book[x]:
            word_index.append(word)
            word_dict.append(Emo_book[x])

word_index = np.array(word_index, dtype=str)
word_dict = np.array(word_dict, dtype=int)

print(word_index.shape,word_dict.shape)
np.save("word_index.npy", word_index)
np.save("word_dict.npy", word_dict)

