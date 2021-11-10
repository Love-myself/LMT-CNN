# -*- encoding: utf-8 -*-
import numpy as np
import csv

file = "CBET.csv"
f = open(file, 'r')
d = [each for each in csv.reader(f, delimiter=',')]
del d[0]
cbet_new = open("CBET_new.txt", 'w')
cbet_max = open("CBET_max.txt", 'w')
# anger, fear, joy, love, sadness, surprise, thankfulness, disgust, guilt.
# 0, 1, 2, 3, 4, 5, 7 #no love
L = [0, 1, 2, 4, 5, 7]  # anger, fear, joy, love, sadness, surprise, disgust
R = [0, 5,2,3,4,1]  # anger disgust joy sad surprise fear love
each_size = 8540
cnt = 0
max_len = 0
for line in d:
    if int(cnt/each_size) in L:
        cbet_new.write(line[1] + '\n')
        cbet_max.write(str(R[L.index(int(cnt/each_size))]) + '\n')
        if len(line[1].split()) > max_len:
            max_len = len(line[1].split())
    cnt += 1
print(max_len)
f.close()
cbet_new.close()
cbet_max.close()
