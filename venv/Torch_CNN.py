import torch
import torchvision#数据库模块，含MINIST
import torch.utils.data as Data

BATCH_SIZE = 50
DOWNLOAD_MINIST = False
LR = 0.02
train_data = torchvision.datasets.MNIST(
    root='./minist',
    train=True,#代表可训练的
    transform=torchvision.transforms.ToTensor(),#转化成tensor
    download= DOWNLOAD_MINIST#下载好了不会重复下载，否则联网下载
)

test_data = torchvision.datasets.MNIST(
    root='./minist',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MINIST
)
train_data_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)
test_x = torch.unsqueeze(test_data.test_data,dim = 1).type(torch.FloatTensor)[:2000]/225#size (-1,1,28,28)
test_y = test_data.test_labels[:2000]


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(     #卷积层block使用Sequential
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2#保持尺寸不变，padding = (kernel_size - 1)/2
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.con2 = torch.nn.Sequential(  #第二个卷积层
            torch.nn.Conv2d(16,32,5,1,2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.out = torch.nn.Linear(32*7*7,10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.con2(x)
        x = x.view(x.size(0),-1) #展开成（32*7*7，-1）
        outputs = self.out(x)
        return outputs

if __name__ == '__main__':
    net = Net()

    optim = torch.optim.SGD(net.parameters(), lr=LR)
    loss_func = torch.nn.CrossEntropyLoss()
    for epho in range(50):
        for step, (batch_x, batch_y) in enumerate(train_data_loader):
            prediction = net(batch_x)
            loss = loss_func(prediction, batch_y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            print('epho:',epho,'step:',step)
            break
    test_x = test_x[:10]
    test_y = test_y[:10]
    prediction = net(test_x)
    pred = torch.max(prediction,1)[1].squeeze()
    print('answer',test_y.data.numpy())
    print('prediction',pred.data.numpy())


