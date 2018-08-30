import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

n_data = torch.ones(100, 2)
x0 = torch.normal(2 * n_data, 1)  # mean , std
y0 = torch.ones(100)
x1 = torch.normal(-2 * n_data, 1)
y1 = torch.zeros(100)

x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # 合并tensor，torch.concat((x1,x2),0)
y = torch.cat((y0, y1), ).type(torch.LongTensor)


class Net(torch.nn.Module):
    def __init__(self, n_features, n_hiddens, n_outputs):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_features, n_hiddens)
        self.output = torch.nn.Linear(n_hiddens, n_outputs)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.output(x)
        return x


net = Net(2, 10, 2)

optimiter = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()
plt.ion()
for t in range(100):
    out = net(x)
    loss = loss_func(out, y)
    optimiter.zero_grad()
    loss.backward()
    optimiter.step()

    if t % 5 == 0:
        plt.cla()
        prediction = torch.max(F.softmax(out),1)[1]#max返回值包括两部分，最大值和最大值的下标
        pred_y = prediction.data.numpy().squeeze()#指定维度，如果维度为1，去掉，否则不变
        target_y = y.data.numpy()
        accuracy = sum(target_y == pred_y)/200
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = sum(pred_y == y) / 200.  # 预测中有多少和真实值一样
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
plt.ioff()
plt.show()
