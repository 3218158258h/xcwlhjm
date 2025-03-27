import sys
import os
from torch.utils.data import dataloader
import tqdm
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms

torch.cuda.set_device(0) #设置使用的GPU是第0号GPU
batch_size = 1

#定义字符类别，包括数字、字母和中文省份缩写
numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphbets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                 'U', 'V', 'W', 'X', 'Y', 'Z']
chinese = ['zh_cuan', 'zh_e', 'zh_gan', 'zh_gan1', 'zh_gui', 'zh_gui1', 'zh_hei', 'zh_hu', 'zh_ji', 'zh_jin',
                'zh_jing', 'zh_jl', 'zh_liao', 'zh_lu', 'zh_meng', 'zh_min', 'zh_ning', 'zh_qing', 'zh_qiong',
                'zh_shan', 'zh_su', 'zh_sx', 'zh_wan', 'zh_xiang', 'zh_xin', 'zh_yu', 'zh_yu1', 'zh_yue', 'zh_yun',
                'zh_zang', 'zh_zhe']

#定义CNN
class char_cnn_net(nn.Module):
    def __init__(self):
        super().__init__()

#定义卷积层和全连接层
        self.conv = nn.Sequential(
            nn.Conv2d(1,64,3,1,1),
            nn.PReLU(),
            nn.Conv2d(64,16,3,1,1),
            nn.PReLU(),
            nn.Conv2d(16,4,3,1,1),
            nn.PReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(1600, 512),
            nn.PReLU(),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.Linear(256,67)
        )

#前向传播方法
    def forward(self, x):
        y = self.conv(x).reshape(batch_size, -1,)
        # print(y.shape)
        return self.fc(y)

#定义自定义数据集类
class CharPic(data.Dataset):

    def list_all_files(self, root):
        files = []
        list = os.listdir(root)
        for i in range(len(list)):
            element = os.path.join(root, list[i])
            if os.path.isdir(element):
                files.extend(self.list_all_files(element))
            elif os.path.isfile(element):
                files.append(element)
        return files

    def __init__(self, root):
        super().__init__()
        if not os.path.exists(root):
            raise ValueError('没有找到文件夹')
        files = self.list_all_files(root)

        self.X = []
        self.y = []
        self.dataset = numbers + alphbets + chinese

        for file in files:
            src_img = cv2.imread(file, cv2.COLOR_BGR2GRAY)
            if src_img.ndim == 3:
                continue
            resize_img = cv2.resize(src_img, (20, 20))
            self.X.append(resize_img)

            dir = os.path.dirname(file)
            dir_name = os.path.split(dir)[-1]

            # vector_y = [0 for i in range(len(self.dataset))]
            index_y = self.dataset.index(dir_name)
            # vector_y[index_y] = 1
            self.y.append([index_y])

        self.X = np.array(self.X)
        self.y = np.array(self.y)

    #定义获取数据集中单个样本的方法
    def __getitem__(self, index):
        tf = transforms.ToTensor()
        # print(torch.Tensor(self.y[index]).shape)
        #把图像转换为Tensor并返回
        return tf(self.X[index]), torch.LongTensor(self.y[index])

    #定义返回数据集大小的方法
    def __len__(self) -> int:
        return len(self.X)

#定义权重初始化函数
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(mean=0.0, std=0.1)
        m.bias.data.fill_(0)

#定义训练函数
def train(epoch, lr):
    model.train()

    #设置loss函数
    criterion = nn.CrossEntropyLoss()
    loss_history = []

    #开始遍历dataloader中的数据。batch_idx是批次的索引，input是批次中的输入数据，target是对应的标签
    for batch_idx, (input, target) in enumerate(train_loader):
        input, target = input.cuda(), target.cuda()
        input, target = Variable(input), Variable(target).reshape(batch_size, )

        #创建Adam优化器，用于后续的参数更新
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer.zero_grad()

        output = model(input)

        #执行前向传播，计算模型对当前批次数据的输出。算出输出和目标真值的损失值
        loss = criterion(output, target)

        # 执行反向传播，计算损失值相对于模型参数的梯度
        loss.backward()

        #如果损失大于历史损失，学习率乘以0.95，衰减学习率
        if loss_history and loss_history[-1] < loss.data:
            lr *= 0.95
        loss_history.append(loss.data)

        #计算出的梯度更新模型的参数
        optimizer.step()

        #监控训练过程
        if batch_idx % 12000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(input), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))

#定义评估模型准确率的函数（计算模型在训练集上的准确率，并在必要时保存模型）
def get_accuracy(model, train_model_path):
    tot = len(train_loader.dataset)
    right = 0

    with torch.no_grad():
        for (input, target) in train_loader:
            input, target = input.cuda(), target.cuda()
            output = model(input)

            for idx in range(len(output)):
                _, predict = torch.topk(output[idx], 1)
                if predict == target[idx]:
                    right += 1

        acc = right / tot
        print('accuracy : %.3f' % acc)
        
        global best_acc
        if acc > best_acc:
            best_acc = acc
            torch.save(model, train_model_path)


if __name__ == '__main__':

    #设置数据目录和模型保存路径
    data_dir = '../images/cnn_char_train'
    train_model_path = 'char.pth'

    #构建定义的CNN模型，并利用GPU和权重进行训练
    model = char_cnn_net()
    #model = torch.load(train_model_path)
    model = model.cuda()
    model.apply(weights_init)

    print("Generate Model.")

    batch_size = 1

    #初始化数据集和DataLoader
    dataset = CharPic(data_dir)
    train_loader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size,
                    num_workers=14, pin_memory=True, drop_last=True)

    global best_acc
    best_acc = 0.0

    #开始训练
    for epoch in range(100):
        lr = 0.001
        train(epoch, lr)
        get_accuracy(model, train_model_path)

    #保存最终训练的模型
    torch.save(model, train_model_path)

    print("Finish Training")
