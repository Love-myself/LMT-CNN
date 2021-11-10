# -*- encoding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score
import warnings
import critic as ct
import scipy.io as sio

warnings.filterwarnings("ignore")


def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)  # m.weight.data是卷积核参数


class TextCNN(nn.Module):
    def __init__(self, vec_dim, filter_num, sentence_max_size, label_size, kernel_list):
        """
        :param vec_dim: 词向量的维度  1x300
        :param filter_num: 每种卷积核的个数 100
        :param sentence_max_size:一个句子的包含的最大的词数量 15
        :param label_size:标签个数，全连接层输出的神经元数量=标签个数 6
        :param kernel_list:卷积核列表 3,4,5
        """
        super(TextCNN, self).__init__()
        chanel_num = 1
        # nn.ModuleList相当于一个卷积的列表，相当于一个list
        # nn.Conv2d的输入是个四维的tensor，每一位分别代表batch_size、channel、length、width
        # nn.MaxPool2d()是最大池化，此处对每一个向量取最大值，所有kernel_size为卷积操作之后的向量维度
        self.convs = nn.ModuleList([nn.Sequential(
            nn.Conv2d(chanel_num, filter_num, (kernel, vec_dim)),  # 1, 100, (3, 300)
            nn.ReLU(),
            # 经过卷积之后，得到一个维度为sentence_max_size - kernel + 1的一维向量
            nn.MaxPool2d((sentence_max_size - kernel + 1, 1))
        ) for kernel in kernel_list])

        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(filter_num * len(kernel_list), label_size),
            nn.Softmax(dim=1),
        )
        # # 全连接层，因为有6个标签
        # self.fc = nn.Linear(filter_num * len(kernel_list), label_size)
        # # dropout操作，防止过拟合
        # self.dropout = nn.Dropout(0.5)
        # # 分类
        # self.sm = nn.Softmax(dim=1)  # dim=0
        self.convs.apply(gaussian_weights_init)
        self.fc.apply(gaussian_weights_init)

    def forward(self, x):
        # Conv2d的输入x是个四维的tensor，每一位分别代表batch_size、channel、length、width
        in_size = x.size(0)  # x.size(0)，表示的是输入x的batch_size
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)  # torch.cat列拼接
        # s1 = out.shape  #Test torch.Size([25, 300, 1, 1])
        out = out.view(in_size, -1)  # 设经过max pooling之后，有output_num个数，将out变成(batch_size,output_num)，-1表示自适应
        # s = out.shape   Test torch.Size([25, 300])
        out = self.fc(out)  # nn.Linear接收的参数类型是二维的tensor(batch_size,output_num),一批有多少数据，就有多少行
        return out


class KL_Loss(nn.Module):
    def __init__(self):
        super(KL_Loss, self).__init__()

    def forward(self, pred, label):
        loss = 0.0
        # pe = torch.sum(torch.exp(pred), axis=1)
        for i in range(pred.shape[0]):
            # portion = torch.log(torch.div(torch.exp(pred[i]), pe[i]))
            portion = torch.log(pred[i])
            loss = torch.add(loss, torch.sum(label[i] * portion))
        loss = torch.div(loss, -pred.shape[0])
        return loss


class My_CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(My_CrossEntropyLoss, self).__init__()

    def forward(self, pred, label):
        loss = 0.0
        # pe = torch.sum(torch.exp(pred), axis=1)
        for i in range(pred.shape[0]):
            # portion = torch.log(torch.div(torch.exp(pred[i]), pe[i]))
            portion = torch.log(pred[i])
            loss = torch.add(loss, -portion[label[i]])
        loss = torch.div(loss, pred.shape[0])
        return loss


def compute_classification(Test_loader):
    model = TextCNN(vec_dim=embedding_dim, filter_num=filter_num, sentence_max_size=sentence_max_size,
                    label_size=label_size, kernel_list=kernel_list)
    model.load_state_dict(torch.load('record3/model.pth'))
    model.eval()
    y_true = np.array([], dtype=int)
    y_pred = np.array([], dtype=int)
    rd = np.array([[]], dtype=float)
    pd = np.array([[]], dtype=float)
    feature = np.array([[]], dtype=float)
    Acc = 0.0
    with torch.no_grad():
        for i, data in enumerate(Test_loader):
            batch_pred = model(data[0])  # batch_pred 预测值
            tem_data = torch.argmax(data[1], axis=1)
            if i == 0:
                feature = data[0]
                rd = data[1]
                pd = batch_pred.data.numpy()
            else:
                feature = np.concatenate((feature, data[0]))
                rd = np.concatenate((rd, data[1]))
                pd = np.concatenate((pd, batch_pred.data.numpy()))
            y_true = np.concatenate([y_true, tem_data.numpy()])
            y_pred = np.concatenate([y_pred, np.argmax(batch_pred.data.numpy(), axis=1)])
            Acc += np.sum(np.argmax(batch_pred.data.numpy(), axis=1) == tem_data.numpy())
            # print(data[0],data[1])

    F = "pre_" + str(k) + ".mat"
    sio.savemat(F, {'trainFeature': feature, 'rd': rd, 'pd': pd})
    Cos = ct.Cosine(rd, pd)
    Int = ct.Intersection(rd, pd)
    Sre = ct.Srensen(rd, pd)
    Squ = ct.Squared(rd, pd)
    K_L = ct.K_L(rd, pd)
    Euc = ct.Euclidean(rd, pd)

    Acc /= Test_set.__len__()
    F1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    result_name = open("result/new_cnn_0.txt", "a")
    print("Euc:%3.6f Sre:%3.6f Squ:%3.6f K_L:%3.6f Cos:%3.6f Int:%3.6f Pre:%3.6f Rec:%3.6f F1:%3.6f Acc:%3.6f"
          % (Euc, Sre, Squ, K_L, Cos, Int, precision, recall, F1, Acc), file=result_name)
    print("Pre:%3.6f Rec:%3.6f F1:%3.6f Acc:%3.6f" % (precision, recall, F1, Acc))
    # print("Pre:%3.6f Rec:%3.6f F1:%3.6f Acc:%3.6f" % (precision, recall, F1, Acc), file=result_name)


def train_textcnn_model(model, train_loader, val_loader, num_epoch, lr, lamda):
    print("begin training...")
    best_acc = 0.0
    min_loss = 10.0
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)  # 优化器
    # loss1 = nn.CrossEntropyLoss()
    loss1 = My_CrossEntropyLoss()  # 损失函数1
    loss2 = KL_Loss()  # 损失函数2
    for epoch in range(num_epoch):  # 迭代循环
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0
        model.train()  # 将模型设置为训练模式
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()  # 优化器参数初始化
            batch_pred = model(data[0])  # 一个batch_size数据量的模型预测值
            tem_data = torch.argmax(data[1], axis=1)  # 计算一个batch_size数据真实情感分布的最大值索引
            batch_loss = (1.0 - lamda) * loss1(batch_pred, tem_data) + lamda * loss2(batch_pred, data[1])  # 计算损失
            batch_loss.backward()  # 根据损失反向传播更新参数
            optimizer.step()
            train_acc += np.sum(np.argmax(batch_pred.data.numpy(), axis=1) == tem_data.numpy())  # 预测准确值得数量
            train_loss += batch_loss.item()

        model.eval()  # 将模型设置为测试模式
        for i, data in enumerate(val_loader):
            batch_pred = model(data[0])
            tem_data = torch.argmax(data[1], axis=1)
            batch_loss = (1.0 - lamda) * loss1(batch_pred, tem_data) + lamda * loss2(batch_pred, data[1])
            val_acc += np.sum(np.argmax(batch_pred.data.numpy(), axis=1) == tem_data.numpy())
            val_loss += batch_loss.item()
        train_acc = train_acc / train_set.__len__()
        val_acc = val_acc / val_set.__len__()
        print('[%03d/%03d] Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f Loss: %3.6f' \
              % (epoch + 1, num_epoch, train_acc, train_loss, val_acc, val_loss))

    model_name = r'record3/model' + '.pth'
    torch.save(model.state_dict(), model_name)
    print('Finished Training')


def Test_textcnn_model(Test_loader, lamda):
    model = TextCNN(vec_dim=embedding_dim, filter_num=filter_num, sentence_max_size=sentence_max_size,
                    label_size=label_size, kernel_list=kernel_list)
    model_name = r'record3/model' + '.pth'
    model.load_state_dict(torch.load(model_name))
    model.eval()  # 必备，将模型设置为训练模式
    Test_acc = 0.0
    for i, data in enumerate(Test_loader):
        batch_pred = model(data[0])
        tem_data = torch.argmax(data[1], axis=1)
        Test_acc += np.sum(np.argmax(batch_pred.data.numpy(), axis=1) == tem_data.numpy())
    Test_acc = Test_acc / Test_set.__len__()
    print('Accuracy of the network on Test set: %3.6f' % Test_acc)


if __name__ == '__main__':

    for k in range(10):
        print("this is the " + str(k) + "th cross:")
        train_set = torch.load('save3/train_set' + str(k) + '.pth')  # 训练数据
        train_label = torch.load('save3/train_label' + str(k) + '.pth')  # 训练数据标签
        val_set = torch.load('save3/val_set' + str(k) + '.pth')  # 验证数据
        val_label = torch.load('save3/val_label' + str(k) + '.pth')  # 验证数据标签
        Test_set = torch.load('save3/test_set' + str(k) + '.pth')  # 测试数据
        Test_label = torch.load('save3/test_label' + str(k) + '.pth')  # 测试集标签

        train_set = TensorDataset(train_set, train_label)  # 训练数据装成TensorDataset
        val_set = TensorDataset(val_set, val_label)  # 验证数据装成TensorDataset
        Test_set = TensorDataset(Test_set, Test_label)  # 测试数据装成TensorDataset

        Train_DataLoader = DataLoader(train_set, batch_size=50, shuffle=True)  # 使用DataLoader组织数据, 设置batch_size, 打乱数据
        Val_DataLoader = DataLoader(val_set, batch_size=50, shuffle=True)
        Test_DataLoader = DataLoader(Test_set, batch_size=50, shuffle=True)

        embedding_dim = 300  # 词向量维度
        filter_num = 100  # 卷积器数量
        sentence_max_size = 15  # 句子最大长度
        label_size = 6  # 分类输出数量
        kernel_list = [3, 4, 5]  # 3种卷积器大小，每种100个
        num_epoch = 200  # 设置迭代次数
        lr = 0.02  # 学习率

        model = TextCNN(vec_dim=embedding_dim, filter_num=filter_num, sentence_max_size=sentence_max_size,
                        label_size=label_size, kernel_list=kernel_list)  # text_cnn模型
        lamda = 0.7
        train_textcnn_model(model, Train_DataLoader, Val_DataLoader, num_epoch, lr, lamda)
        Test_textcnn_model(Test_DataLoader, lamda)
        compute_classification(Test_DataLoader)
