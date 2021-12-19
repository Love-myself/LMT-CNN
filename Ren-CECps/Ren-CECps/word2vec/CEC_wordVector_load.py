# -*- encoding: utf-8 -*-
import numpy as np
# import gensim
import torch
import jieba
# word_index = np.load("../data/generatedFile/dictCEC.txt")
# word_dict = np.load("../data/generatedFile/dictCEC.txt")
word_index = np.load("../LB/word_index.npy")
word_dict = np.load("../LB/word_dict.npy")
# 情感词
word_index = word_index.tolist()
word_dict = word_dict.tolist()


def ReadData(path1, path2, path3, path4):   # 训练集文本、情绪分布, 测试集文本、情绪分布
    # train_data = []
    train_y = []
    train_sentence = np.load(path1)
    train_sentence = train_sentence.tolist()

    for line in train_sentence:
        D = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}  # 字典，记录一条语句各类情绪标签数量
    #     # matrix = np.zeros((100, 300), dtype='float16')
    #     a = np.zeros((1, 300))
    #     max1 = 0
    #     list = []
    #     cnt1 = 0
    #     for i in line:
    #         if i == '\n':
    #             continue
    #         try:
    #             b = np.array([model[i]])
    #             a = np.vstack((a, b))
    #             c = np.delete(a, 0, 0)
    #         except KeyError:
    #             b = np.random.uniform(0, 0, 300)
    #             a = np.vstack((a, b))
    #             c = np.delete(a, 0, 0)
    #         # matrix[cnt1] = a
    #         # cnt1 = cnt1 + 1
    #         # print(str(i) + "not in vocabulary")
    #     if c.shape[0] > max1:
    #         max1 = c.shape[0]
    #     if c.shape[0] < 100:
    #         dis = 100 - c.shape[0]
    #         b = np.zeros((dis, 300))
    #         c = np.vstack((c, b))
    #     list.append(c)
    # train_data = np.array(list, dtype='float16')
    # train_data = train_data.reshape(31534, 1, 100, 300)
    # data = np.array(matrix)
        # cnt = 0
        # for word in line.split():
        #     try:
        #         embedding = np.array([model[word]])
        #     except Exception:  # 如果单词不在语料库中，在[-1,1]中随机定义
        #         embedding = np.random.uniform(0,0, 300)
        #     matrix[cnt] = embedding
        #     cnt += 1
        # train_data.append(matrix.reshape(1, 100, 300))

        # jieba
        seg = jieba.cut(line)
        line_fenci = " ".join(seg)
        line_fenci = line_fenci.split()

        for word in line_fenci:
            if word in word_index:
                pos = word_index.index(word)
                pl = [i for i, w in enumerate(word_dict[pos]) if w == 1]
                for i in pl:
                    if i != 8:
                        D[i] += 1
        D_sum = 0  # 一条语句情感词数量总和
        for u in range(8):
            # if D[u] != 0:
            # D_cnt.append(u)
            D_sum += D[u]
            #print(D_sum)
        matrix1 = np.zeros((8, 1), dtype='float16')
        cnt1 = 0

        for i, w in enumerate(D):
            if D_sum == 0:
               # matrix1 = [[1/6] , [1/6] , [1/6] , [1/6] , [1/6] , [1/6]]
               matrix1 = [[0], [0], [0], [0], [0], [0], [0], [0]]
               #  Joy, Hate, Love, Sorrow, Anxiety, Surprise, Anger, Expect
            else:
                if w == 0:
                    # word = 'Joy'
                    embedding1 =  D[w] / D_sum
                    matrix1[cnt1] = embedding1
                    cnt1 += 1
                elif w == 1:
                    # word = 'Hate'
                    embedding1 = D[w] / D_sum
                    matrix1[cnt1] = embedding1
                    cnt1 += 1
                elif w == 2:
                    # word = 'Love'
                    embedding1 = D[w] / D_sum
                    matrix1[cnt1] = embedding1
                    cnt1 += 1
                elif w == 3:
                    # word = 'Sorrow'
                    embedding1 = D[w] / D_sum
                    matrix1[cnt1] = embedding1
                    cnt1 += 1
                elif w == 4:
                    # word = 'Anxiety'
                    embedding1 = D[w] / D_sum
                    matrix1[cnt1] = embedding1
                    cnt1 += 1
                elif w == 5:
                    # word = 'Surprise'
                    embedding1 = D[w] / D_sum
                    matrix1[cnt1] = embedding1
                    cnt1 += 1
                elif w == 6:
                    # word = 'Anger'
                    embedding1 = D[w] / D_sum
                    matrix1[cnt1] = embedding1
                    cnt1 += 1
                elif w == 7:
                    # word = 'Expect'
                    embedding1 = D[w] / D_sum
                    matrix1[cnt1] = embedding1
                    cnt1 += 1

        # print(matrix1.shape, matrix1)
        train_y.append(matrix1)


    # train_data = np.array(train_data, dtype='float16')
    # data_size = train_data.shape[0]
    # shuffled_index = np.random.permutation(data_size)
    # train_data = train_data[shuffled_index]

    train_label_data = np.load(path2)
    # train_label_data = train_label_data[shuffled_index]
    train_label_single_data = np.argmax(train_label_data, axis=1)

    train_y = np.array(train_y, dtype='float16')
    # train_y = np.array(train_y)[shuffled_index]


    td = train_label_single_data
    eight_motion = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: []}
    for i in range(td.shape[0]):
        if td[i] != 8:
            eight_motion[td[i]].append(i)
    val_pos = []
    for i in range(8):
        ln = round(len(eight_motion[i]) * 0.1)
        for j in eight_motion[i][:ln]:
            val_pos.append(j)
        #print(ln)
    train_pos = [w for w in range(31534) if w not in val_pos]
    #print(len(val_pos))
    val_pos = np.array(val_pos, dtype=int)
    train_pos = np.array(train_pos, dtype=int)


    # val_set = train_data[val_pos]  # 验证集
    # train_set = train_data[train_pos]  # 训练集
    #
    # val_label = train_label_data[val_pos]   # 验证集标签
    # train_label = train_label_data[train_pos]  # 训练集标签
    #
    # val_label_single = train_label_single_data[val_pos]   # 验证集标签最大下标
    # train_label_single = train_label_single_data[train_pos]  # 训练集标签最大下标

    val_y = train_y[val_pos]
    train_y = train_y[train_pos]

    # val_set = torch.FloatTensor(val_set)   # 变为张量形式
    # train_set = torch.FloatTensor(train_set)
    # val_label = torch.FloatTensor(val_label)
    # train_label = torch.FloatTensor(train_label)
    # val_label_single = torch.LongTensor(val_label_single)
    # train_label_single = torch.LongTensor(train_label_single)
    val_y =  torch.FloatTensor(val_y)
    train_y = torch.FloatTensor(train_y)

    test_data = []
    test_y = []
    test_sentence = np.load(path3)
    test_sentence = test_sentence.tolist()

    for line in test_sentence:
        # D = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        # matrix2 = CEC_wordVector_loadnp.zeros((15, 300), dtype=float)
        # cnt = 0
        # for word in line.split():
        #     try:
        #         embedding = np.array([model[word]])
        #
        #     except Exception:  # 如果单词不在语料库中，在[-1,1]中随机定义
        #         embedding = np.random.uniform(0,0,300)
        #         #print(word + '  ')
        #     matrix2[cnt] = embedding
        #     cnt += 1

        '''D = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}  # 字典，记录一条语句各类情绪标签数量
        matrix2 = np.zeros((100, 300), dtype='float16')
        a = np.zeros((1, 300))
        max1 = 0
        cnt1 = 0
        for i in line:
            if i == '\n':
                continue
            try:
                b = np.array([model[i]])
                a = np.vstack((a, b))
                c = np.delete(a, 0, 0)
            except KeyError:
                b = np.random.uniform(0, 0, 300)
                a = np.vstack((a, b))
                c = np.delete(a, 0, 0)
            matrix2[cnt1] = b
            cnt1 = cnt1 + 1
            # print(str(i) + "not in vocabulary")
        if c.shape[0] > max1:
            max1 = c.shape[0]
        if c.shape[0] < 100:
            dis = 100 - c.shape[0]
            d = np.zeros((dis, 300))
            c = np.vstack((c, d))
        test_data.append(matrix2.reshape(1, 100, 300))'''
        # a = np.zeros((1, 300))
        # max1 = 0
        # list = []
        # cnt1 = 0
        # for i in line:
        #     if i == '\n':
        #         continue
        #     try:
        #         b = np.array([model[i]])
        #         a = np.vstack((a, b))
        #         c = np.delete(a, 0, 0)
        #     except KeyError:
        #         b = np.random.uniform(0, 0, 300)
        #         a = np.vstack((a, b))
        #         c = np.delete(a, 0, 0)
        #     # matrix[cnt1] = a
        #     # cnt1 = cnt1 + 1
        #     # print(str(i) + "not in vocabulary")
        # if c.shape[0] > max1:
        #     max1 = c.shape[0]
        # if c.shape[0] < 100:
        #     dis = 100 - c.shape[0]
        #     b = np.zeros((dis, 300))
        #     c = np.vstack((c, b))
        # list.append(c)
        # data = np.array(list, dtype='float16')
        # test_data.append(data.reshape(100, 300))

        seg = jieba.cut(line)
        line_fenci = " ".join(seg)
        line_fenci = line_fenci.split()

        for word in line_fenci:
            if word in word_index:
                pos = word_index.index(word)
                pl = [i for i, w in enumerate(word_dict[pos]) if w == 1]
                for i in pl:
                    if i != 8:
                        D[i] += 1
        D_sum = 0  # 一条语句情感词数量总和
        for u in range(8):
            # if D[u] != 0:
            # D_cnt.append(u)
            D_sum += D[u]
            #print(D_sum)
        matrix3 = np.zeros((8, 1), dtype='float16')
        cnt1 = 0
        for i, w in enumerate(D):
            if D_sum == 0:
               #matrix3 = [[1/6] , [1/6] , [1/6] , [1/6] , [1/6] , [1/6]]
               matrix3 = [[0], [0], [0], [0], [0], [0], [0], [0]]
            else:
                if w == 0:
                    #word = 'anger'
                    embedding1 =  D[w] / D_sum
                    matrix3[cnt1] = embedding1
                    cnt1 += 1
                elif w == 1:
                    #word = 'disgust'
                    embedding1 = D[w] / D_sum
                    matrix3[cnt1] = embedding1
                    cnt1 += 1
                elif w == 2:
                   # word = 'joy'
                    embedding1 = D[w] / D_sum
                    matrix3[cnt1] = embedding1
                    cnt1 += 1
                elif w == 3:
                    #word = 'sad'
                    embedding1 = D[w] / D_sum
                    matrix3[cnt1] = embedding1
                    cnt1 += 1
                elif w == 4:
                    #word = 'surprise'
                    embedding1 = D[w] / D_sum
                    matrix3[cnt1] = embedding1
                    cnt1 += 1
                elif w == 5:
                    #word = 'fear'
                    embedding1 = D[w] / D_sum
                    matrix3[cnt1] = embedding1
                    cnt1 += 1
                elif w == 6:
                    #word = 'fear'
                    embedding1 = D[w] / D_sum
                    matrix3[cnt1] = embedding1
                    cnt1 += 1
                elif w == 7:
                    #word = 'fear'
                    embedding1 = D[w] / D_sum
                    matrix3[cnt1] = embedding1
                    cnt1 += 1
        test_y.append(matrix3)

    # test_set = np.array(test_data, dtype='float16')
    # test_label = np.load(path4)
    # test_label_single = np.argmax(test_label, axis=1)
    test_y = np.array(test_y, dtype='float16')
    #print(test_y)
    #test_y = torch.LongTensor(test_y)

    '''
    train_label_data = np.load(path2)
    train_label_data = train_label_data[shuffled_index]
    train_label_single_data = np.argmax(train_label_data, axis=1)

    train_y = np.array(train_y, dtype='float16')
    train_y = np.array(train_y)[shuffled_index]
    '''

    # test_set = torch.FloatTensor(test_set)
    # test_label = torch.FloatTensor(test_label)
    # test_label_single = torch.LongTensor(test_label_single)
    test_y = torch.FloatTensor(test_y)
    print('test_y: ', test_y)



    # return train_set, train_label, train_label_single,train_y, val_set, val_label, val_label_single,val_y, test_set, test_label, test_label_single,test_y
    return train_y, val_y, test_y

if __name__ == "__main__":
    print("ReadData...")
    # model = gensim.models.KeyedVectors.load_word2vec_format('sgns_weibo_word_char.txt', binary=False)
    for k in range(10):
        file1name = "../10-cross/cross" + str(k) + "/train_data.npy"  # cbet 句子训练集
        file2name = "../10-cross/cross" + str(k) + "/train_label.npy"  # cbet 训练集情绪分布
        file3name = "../10-cross/cross" + str(k) + "/test_data.npy"  # cbet 句子测试集
        file4name = "../10-cross/cross" + str(k) + "/test_label.npy"  # cbet 测试集情绪分布
        print("Data is coming!")
        # train_set, train_label, train_label_single,train_y,  val_set, val_label, val_label_single,val_y, test_set, test_label, test_label_single ,test_y  = ReadData(
        #     file1name, file2name, file3name, file4name)
        train_y, val_y, test_y = ReadData(
            file1name, file2name, file3name, file4name)
        #print(test_set, test_label, test_label_single ,test_y)

        # x_train = torch.save(train_set, 'save3/train_set' + str(k) + '.pth')
        # x_label = torch.save(train_label, 'save3/train_label' + str(k) + '.pth')
        # x_label_max = torch.save(train_label_single, 'save3/train_label_single' + str(k) + '.pth')
        x_y = torch.save(train_y, '../save/train_y' + str(k) + '.pth')

        # y_val = torch.save(val_set, 'save3/val_set' + str(k) + '.pth')
        # y_label = torch.save(val_label, 'save3/val_label' + str(k) + '.pth')
        # y_label_max = torch.save(val_label_single, 'save3/val_label_single' + str(k) + '.pth')
        y_y = torch.save(val_y, '../save/val_y' + str(k) + '.pth')

        # z_val = torch.save(test_set, 'save3/test_set' + str(k) + '.pth')
        # z_label = torch.save(test_label, 'save3/test_label' + str(k) + '.pth')
        # z_label_max = torch.save(test_label_single, 'save3/test_label_single' + str(k) + '.pth')
        z_y = torch.save(test_y, '../save/test_y' + str(k) + '.pth')

        # print(train_set.shape)
        # print(train_label.shape)
        # print(train_label_single.shape)
        print("train_y.shape:")
        print(train_y.shape)
        # print(val_set.shape)
        # print(val_label.shape)
        # print(val_label_single.shape)
        print('val_y.shape:')
        print(val_y.shape)
        # print(test_set.shape)
        # print(test_label.shape)
        # print(test_label_single.shape)
        print('val_y.shape:')
        print(test_y.shape)
