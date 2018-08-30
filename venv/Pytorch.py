import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.autograd import Variable

# tensor和array的不断转化
a = [[12,24],[56,78]]
a = np.array(a)
b = torch.from_numpy(a)
c = b.numpy()
d = torch.FloatTensor(a)
print(
    '\na', a,
    '\nb', b,
    '\nc', c,
    '\nd', d
)

# torch中的变量
tensor = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])#用一个list初始化一个tensor
variable = Variable(tensor, requires_grad=True)  # 用一个tensor来初始化变量
v_out = torch.mean(variable * variable)
v_out.backward()  # 模拟反向误差传递
print(variable.grad)  # 计算梯度

# 获取variable中的数据
print(variable)
print(variable.data.numpy())

# 画散点图
x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # 添加一个维度
y = x.pow(2) + 0.2 * torch.rand(x.size())
plt.scatter(x.numpy(), y.numpy())
plt.show()

# 以下代码为建立一个神经网络



# 以下代码为建立一个神经网络
import torch
import torch.nn.functional as F
from torch.autograd  import Variable
import matplotlib.pyplot as plt




x = torch.unsqueeze(torch.linspace(-1,1,100),dim = 1)#变成无数个样本

y = x.pow(2) + 0.2*torch.rand(x.size())






class Net(torch.nn.Module):  # 继承自torch.nn.Moudle
    def __init__(self, n_features, n_hiddens, n_outputs):  # 确定各层的层属性
        super(Net, self).__init__()

        self.hidden = torch.nn.Linear(n_features, n_hiddens)
        self.predict = torch.nn.Linear(n_hiddens, n_outputs)

    def forward(self, x):  # 通过激活函数搭建层与层的联系
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x


net = Net(1, 10, 1)
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)  # 获得优化器
loss_func = torch.nn.MSELoss()  # 获得损失函数，是误差均方差
plt.ion()

for t in range(100):
    prediction = net(x)
    loss = loss_func(prediction, y)
    loss.backward()  # 损失函数反向传递
    optimizer.zero_grad()  # 清空上一步残余的更新参数值
    optimizer.step()  # 将参数更新施加到net的parametres上
    if t % 5 == 0:
        plt.cla()#清空axis
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'Loss = %.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)#暂停


plt.ioff()
plt.show()