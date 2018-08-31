import torch
import torch.utils.data  as Data
BATCH_SIZE = 5
torch.manual_seed(1)
x = torch.linspace(-1,1,10)
y = torch.linspace(-2,2,10)

data_set = Data.TensorDataset(x,y)#用tensor初始化一个数据集
loader = Data.DataLoader(
    dataset=data_set,           #数据集
    batch_size=BATCH_SIZE,      #批量处理数目
    shuffle=True,               #打乱顺序
    num_workers=2,              #分在两个线程中
)

def show_loader():
    for epoch in range(3):
        for step, (batch_x, batch_y) in enumerate(loader):
            print('Epoch:', epoch, '|step:', step, '|batch_x:', batch_x.data.numpy(), '|batch_y:', batch_y.data.numpy())

if __name__ == '__main__':
    show_loader()