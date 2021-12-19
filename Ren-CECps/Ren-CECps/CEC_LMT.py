import numpy as np
import torch
import torch.nn as nn
import scipy.io as sio
import critic as ct
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)  # m.weight.data是卷积核参数


class KL_Loss(nn.Module):
    def __init__(self):
        super(KL_Loss, self).__init__()

    def forward(self, pred, label):
        loss = 0.0
        for i in range(pred.shape[0]):
            portion = torch.log(pred[i])
            loss = torch.add(loss, torch.sum(label[i] * portion))
        loss = torch.div(loss, -pred.shape[0])
        return loss


class My_CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(My_CrossEntropyLoss, self).__init__()

    def forward(self, pred, label):
        loss = 0.0
        for i in range(pred.shape[0]):
            portion = torch.log(pred[i])
            loss = torch.add(loss, -portion[label[i]])
        loss = torch.div(loss, pred.shape[0])
        return loss


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
            nn.Linear(filter_num * len(kernel_list) + label_size, label_size),
            nn.Softmax(dim=1),
        )
        self.convs.apply(gaussian_weights_init)
        self.fc.apply(gaussian_weights_init)

    def forward(self, x, y):
        in_size = x.size(0)  # 输入x的batch_size
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)  # 将out从list转成[batch,300,1,1]
        out = out.view(in_size, -1)  # 将out变成(batch_size,output_num) [batch,300]
        y = np.reshape(y, (in_size, label_size))
        out = torch.cat((out, torch.from_numpy(y)), dim=1)  # [batch,306]
        out = self.fc(out)
        return out


def compute_classification(test_loader):
    model = TextCNN(vec_dim=embedding_dim, filter_num=filter_num, sentence_max_size=sentence_max_size,
                    label_size=label_size, kernel_list=kernel_list)
    model.load_state_dict(torch.load("record/LMTmodel.pth"))
    model.eval()
    single_true = np.array([], dtype=int)
    single_pred = np.array([], dtype=int)
    rd = np.array([[]], dtype=float)
    pd = np.array([[]], dtype=float)
    feature = np.array([[]], dtype=float)
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            y = data[2].numpy() * seta
            batch_pred = model(data[0], y)  # batch_pred 预测值
            tem_data = torch.argmax(data[1], axis=1)
            if i == 0:
                feature = data[0]
                rd = data[1]
                pd = batch_pred.data.numpy()
            else:
                feature = np.concatenate((feature, data[0]))
                rd = np.concatenate((rd, data[1]))
                pd = np.concatenate((pd, batch_pred.data.numpy()))
            single_true = np.concatenate([single_true, tem_data.numpy()])
            single_pred = np.concatenate([single_pred, np.argmax(batch_pred.data.numpy(), axis=1)])
    # print(feature.shape,rd.shape,pd.shape)
    F = 'pre/LMTpre' + str(k) + '.mat'
    sio.savemat(F, {'trainFeature': feature, 'rd': rd, 'pd': pd})
    Cos = ct.Cosine(rd, pd)
    Int = ct.Intersection(rd, pd)
    Sre = ct.Srensen(rd, pd)
    Squ = ct.Squared(rd, pd)
    K_L = ct.K_L(rd, pd)
    Euc = ct.Euclidean(rd, pd)
    Acc = accuracy_score(single_true, single_pred)
    F1 = f1_score(single_true, single_pred, average='macro')
    precision = precision_score(single_true, single_pred, average='macro')
    recall = recall_score(single_true, single_pred, average='macro')

    result_name = open("result/LMT/LMT_" + str(seta) + ".txt", 'a')
    print("Euc:%3.6f Sre:%3.6f Squ:%3.6f K_L:%3.6f Cos:%3.6f Int:%3.6f Pre:%3.6f Rec:%3.6f F1:%3.6f Acc:%3.6f"
          % (Euc, Sre, Squ, K_L, Cos, Int, precision, recall, F1, Acc), file=result_name)
    # print("Pre:%3.6f Rec:%3.6f F1:%3.6f Acc:%3.6f" % (precision, recall, F1, Acc))
    print("Euc:%3.6f Sre:%3.6f Squ:%3.6f K_L:%3.6f Cos:%3.6f Int:%3.6f Pre:%3.6f Rec:%3.6f F1:%3.6f Acc:%3.6f"
          % (Euc, Sre, Squ, K_L, Cos, Int, precision, recall, F1, Acc))


def train_textcnn_model(model, train_loader, val_loader, num_epoch, lr, lamda):
    print("begin trainning...")
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss1 = My_CrossEntropyLoss()
    loss2 = KL_Loss()
    for epoch in range(num_epoch):
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0
        model.train()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()  # 优化器参数初始化
            # print(data[2][0])
            y = data[2].numpy() * seta
            # print(y[0])
            batch_pred = model(data[0], y)  # 预测的情感分布
            batch_single = torch.argmax(data[1], axis=1)  # 真实情感的最大值索引
            batch_loss = (1.0 - lamda) * loss1(batch_pred, batch_single) + lamda * loss2(batch_pred, data[1])
            batch_loss.backward()
            optimizer.step()
            train_acc += np.sum(np.argmax(batch_pred.data.numpy(), axis=1) == batch_single.numpy())  # 预测准确值得数量
            train_loss += batch_loss.item()

        model.eval()
        for i, data in enumerate(val_loader):
            y = data[2].numpy() * seta
            batch_pred = model(data[0], y)
            batch_single = torch.argmax(data[1], axis=1)
            batch_loss = (1.0 - lamda) * loss1(batch_pred, batch_single) + lamda * loss2(batch_pred, data[1])
            val_acc += np.sum(np.argmax(batch_pred.data.numpy(), axis=1) == batch_single.numpy())  # 预测准确值得数量
            val_loss += batch_loss.item()

        train_acc = train_acc / train_set.__len__()
        val_acc = val_acc / val_set.__len__()
        print('[%03d/%03d] Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f Loss: %3.6f' \
              % (epoch + 1, num_epoch, train_acc, train_loss, val_acc, val_loss))

        print('[%03d/%03d] Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f Loss: %3.6f' \
              % (epoch + 1, num_epoch, train_acc, train_loss, val_acc, val_loss), file=open("epoch.txt", 'a'))

    model_name = r'record/LMTmodel.pth'
    torch.save(model.state_dict(), model_name)
    print("Finished Training")


'''def test_textcnn_model(test_loader, lamda):
    model = TextCNN(vec_dim=embedding_dim, filter_num=filter_num, sentence_max_size=sentence_max_size,
                    label_size=label_size, kernel_list=kernel_list)
    model_name = r'record/LMTmodel.pth'
    model.load_state_dict(torch.load(model_name))
    model.eval()
    test_acc = 0.0
    for i, data in enumerate(test_loader):
        y = data[2].numpy() * seta
        batch_pred = model(data[0], y)
        batch_single = torch.argmax(data[1], axis=1)
        test_acc += np.sum(np.argmax(batch_pred.data.numpy(), axis=1) == batch_single.numpy())
    test_acc = test_acc / test_set.__len__()
    print('Accuracy of the network on Test set: %3.6f' % test_acc)'''


if __name__ == "__main__":
    for k in range(1):
        print("this is the " + str(k) + "th cross:")

        train_data = torch.load("save/train_set" + str(k) + ".pth")
        train_label = torch.load("save/train_label" + str(k) + ".pth")
        train_y = torch.load("save/train_y" + str(k) + ".pth")

        test_data = torch.load("save/test_set" + str(k) + ".pth")
        test_label = torch.load("save/test_label" + str(k) + ".pth")
        test_y = torch.load("save/test_y" + str(k) + ".pth")

        val_data = torch.load("save/val_set" + str(k) + ".pth")
        val_label = torch.load("save/val_label" + str(k) + ".pth")
        val_y = torch.load("save/val_y" + str(k) + ".pth")

        train_set = TensorDataset(train_data, train_label, train_y)  # 训练数据装成TensorDataset
        val_set = TensorDataset(val_data, val_label, val_y)  # 验证数据装成TensorDataset
        test_set = TensorDataset(test_data, test_label, test_y)  # 测试数据装成TensorDataset
        Train_DataLoader = DataLoader(train_set, batch_size=50, shuffle=True)  # 使用DataLoader组织数据, 设置batch_size, 打乱数据
        Val_DataLoader = DataLoader(val_set, batch_size=50, shuffle=True)
        Test_DataLoader = DataLoader(test_set, batch_size=50, shuffle=True)

        embedding_dim = 300  # 词向量维度
        filter_num = 100  # 卷积器数量
        sentence_max_size = 50  # 句子最大长度
        label_size = 8  # 分类输出数量
        kernel_list = [3, 4, 5]  # 3种卷积器大小，每种100个
        num_epoch = 200  # 设置迭代次数
        lr = 0.02  # 学习率

        model = TextCNN(vec_dim=embedding_dim, filter_num=filter_num, sentence_max_size=sentence_max_size,
                        label_size=label_size, kernel_list=kernel_list)  # text_cnn模型
        lamda = 0.7
        # seta = 0.2
        setas = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        for seta in setas:
            train_textcnn_model(model, Train_DataLoader, Val_DataLoader, num_epoch, lr, lamda)
            # test_textcnn_model(Test_DataLoader, lamda)
            compute_classification(Test_DataLoader)
