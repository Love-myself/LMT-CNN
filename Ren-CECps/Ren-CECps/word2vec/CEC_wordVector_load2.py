# *- encoding: utf-8 -*-
import numpy as np
from gensim.models import word2vec
from sklearn.model_selection import train_test_split
import torch

# for i in range(10):
#     file_handle = open('../train_label'+str(i)+'.pth', mode='a')
#     file_handle = open('../test_label'+str(i) + '.pth', mode='a')
#     file_handle = open('../val_set'+str(i) + '.pth', mode='a')
#     file_handle = open('../val_label'+str(i) + '.pth', mode='a')

def w2v(datafile):
    max1 = 0
    sentence = np.load(datafile)
    sentence = sentence.tolist()
    list = []
    for line in sentence:
        a = np.zeros((1, 300))
        for i in line:
            if i == '\n':
                continue
            try:
                b = np.array([model[i]])
                a = np.vstack((a, b))
                c = np.delete(a, 0, 0)
            except KeyError:
                d = np.random.uniform(0, 0, 300)
                a = np.vstack((a, d))
                c = np.delete(a, 0, 0)
                # print(str(i) + "not in vocabulary")
        if c.shape[0] > max1:
            max1 = c.shape[0]
        if c.shape[0] < 50:
            dis = 50 - c.shape[0]
            d = np.zeros((dis, 300))
            c = np.vstack((c, d))
        if c.shape[0] > 50:
            c = c[0:50, 0:300]
        list.append(c)
    print(len(list))
    print(len(list[0]))
    data = np.array(list, dtype="float16")
    print("最大句子长度"+str(max1))
    return data



def cross_save(k):
    file1name = "../10-cross/cross" + str(k) + "/train_data.npy"  # 句子训练集
    file2name = "../10-cross/cross" + str(k) + "/train_label.npy"  # 训练集情绪分布
    file3name = "../10-cross/cross" + str(k) + "/test_data.npy"  # 句子测试集
    file4name = "../10-cross/cross" + str(k) + "/test_label.npy"  # 测试集情绪分布

    train_data = w2v(file1name)
    test_data = w2v(file3name)
    print(train_data.shape)
    print(test_data.shape)
    train_data = train_data.reshape(31534, 1, 50, 300)
    test_data = test_data.reshape(3503, 1, 50, 300)

    print(train_data.shape)
    print(test_data.shape)

    train_label = np.load(file2name)
    test_label = np.load(file4name)

    #证集
    train_label_data = train_label
    test_label_data = test_label
    # train_label_data = train_label_data[shuffled_index]
    train_label_single_data = np.argmax(train_label_data, axis=1)
    test_label_single_data = np.argmax(test_label_data, axis=1)

    # train_y = np.array(train_y, dtype='float16')
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
        # print(ln)
    train_pos = [w for w in range(31534) if w not in val_pos]
    # print(len(val_pos))
    val_pos = np.array(val_pos, dtype=int)
    train_pos = np.array(train_pos, dtype=int)

    val_set = train_data[val_pos]  # 验证集
    train_set = train_data[train_pos]  # 训练集

    val_label = train_label_data[val_pos]   # 验证集标签
    train_label = train_label_data[train_pos]  # 训练集标签

    val_label_single = train_label_single_data[val_pos]   # 验证集标签最大下标
    train_label_single = train_label_single_data[train_pos]  # 训练集标签最大下标
    # train_set, val_set, train_label, val_label = train_test_split(train_data, train_label, test_size=0.10)

    train_set = torch.FloatTensor(train_set)
    train_label = torch.FloatTensor(train_label)
    train_label_single = torch.FloatTensor(train_label_single)
    val_set = torch.FloatTensor(val_set)
    val_label = torch.FloatTensor(val_label)
    val_label_single = torch.FloatTensor(val_label_single)
    test_data = torch.FloatTensor(test_data)
    test_label = torch.FloatTensor(test_label)
    test_label_single_data = torch.FloatTensor(test_label_single_data)
    print('train_set, train_label, train_label_single\n')
    print(train_set.shape)
    print(train_label.shape)
    print(train_label_single.shape)
    print('val_set, val_label, val_label_single\n')
    print(val_set.shape)
    print(val_label.shape)
    print(val_label_single.shape)
    print('test_data, test_label, test_label_single_data\n')
    print(test_data.shape)
    print(test_label.shape)
    print(test_label_single_data.shape)
    print('\n')
    train_set_name = "../save/train_set"+str(k)+".pth"
    train_label_name = "../save/train_label"+str(k)+".pth"
    train_label_single_name = "../save/train_label_single" + str(k) + ".pth"
    torch.save(train_set, train_set_name)
    torch.save(train_label, train_label_name)
    torch.save(train_label_single, train_label_single_name)

    val_set_name = "../save/val_set"+str(k)+".pth"
    val_label_name = "../save/val_label"+str(k)+".pth"
    val_label_single_name = "../save/val_label_single" + str(k) + ".pth"
    torch.save(val_set, val_set_name)
    torch.save(val_label, val_label_name)
    torch.save(val_label_single, val_label_single_name)

    test_set_name = "../save/test_set"+str(k)+".pth"
    test_label_name = "../save/test_label"+str(k)+".pth"
    test_label_single_data_name = "../save/test_label_single_data" + str(k) + ".pth"
    torch.save(test_data, test_set_name)
    torch.save(test_label, test_label_name)
    torch.save(test_label_single_data, test_label_single_data_name)


if __name__ == "__main__":
    print("ReadData...")
    model = word2vec.KeyedVectors.load_word2vec_format('sgns_weibo_word_char.txt', binary=False)

    for i in range(10):
        print("this is the "+str(i)+"th cross:")
        cross_save(i)  # 将每一折W2V,并划分验证集
        print('\n')
