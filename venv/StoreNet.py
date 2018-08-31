import torch

x = torch.unsqueeze(torch.linspace(-1,1,100),dim = 1)
y = x.pow(2) + 0.2 * torch.rand(x.size())

def net():
    net = torch.nn.Sequential(
        torch.nn.Linear(1,10),
        torch.nn.ReLU(),
        torch.nn.Linear(10,1)
    )
    loss_func = torch.nn.MSELoss()
    optim = torch.optim.SGD(net.parameters(),lr=0.02)
    for t in range(100):
        prediction = net(x)
        loss = loss_func(prediction, y)
        loss.backward()
        optim.zero_grad()
        optim.step()
    torch.save(net,'net.pkl')
    torch.save(net.state_dict(),'net_parameters.pkl')
if __name__ == "__main__":
    net()
    net  = torch.load('net.pkl')