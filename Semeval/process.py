# -*- encoding: utf-8 -*-
import nltk
#nltk.download()
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

file = "semeval.txt"
textS = open("semeval_split1.txt", "w")
with open(file) as f:
    lines = f.readlines()
    for line in lines:
        text = nltk.word_tokenize(line)  # 分词
        #text = [w for w in text if w not in stopwords.words("english")]  # 停用词，如is,the等等
        for word in text:
            word = word.lower()  # 转化为小写
            token = ""
            flag = 0
            pos = 0
            '''
            for ch in word:
                if ch.isalnum() is True:
                    flag = 1
                    break
                pos += 1
            '''
            if word[0].isalnum() is True or word[0]=="'":
                x=0
                y=word.__len__()
                if word[0]=="'":
                    x=1
                if word[-1]=="'":
                    y=word.__len__()-1
                textS.write(word[x:y].replace('-', ' ') + ' ')
            '''
                last_pos = pos
                for zh in word[pos:]:
                    #if zh == "'" or zh == "-" or zh == "_":
                    if zh == "-":
                        zh = zh.replace("-"," ")
                    if zh == "'" or zh == "_":
                        pass
                    elif zh.isalnum() is False:
                        break
                    last_pos += 1
                token = word[pos:last_pos]
                textS.write(token + ' ')
            '''
        textS.write('\n')

textS.close()
file2 = "semeval_split1.txt"
textZ = open("semeval_train1.txt", "w")

with open(file2) as f:
    lines = f.readlines()
    print(lines.__len__())
    for line in lines:
        line = line.split()
        if "n't" in line:
            print(line)
        if "not" in line :
            print(line)
        cnt = 0
        for word in line:
            if word == "s":
                del line[cnt]
            elif word == "ve":
                del line[cnt]
            elif word == "n't":
                if line[cnt-1] in ["ca", "do", "did", "could", "wo", "should", "would", "does", "ai"]:
                    line[cnt-1] = line[cnt-1] + "n't"
                del line[cnt]
            elif word == "ll":
                del line[cnt]
            elif word == "m":
                # line[cnt-1] = line[cnt-1] + "'m"
                del line[cnt]
            elif word == "quot":
                del line[cnt]
            cnt += 1
        newL = ' '.join(line)
        textZ.write(newL+'\n')

textZ.close()

