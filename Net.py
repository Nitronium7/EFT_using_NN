import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(21, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 50)
        #self.fc4 = nn.Linear(50, 50)
        self.fc5 = nn.Linear(50, 10)
        self.fc6 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        #x = F.leaky_relu(self.fc4(x))
        x = F.leaky_relu(self.fc5(x))

        x = (self.fc6(x))
        return x

        # y = torch.sigmoid(torch.div(self.fc5(x), div))
        # x = F.softmax(self.fc4(x), dim = 1)
        # x = torch.sigmoid(self.fc4(x))
        # m = nn.Sigmoid()
        # x = m(x)
        # print(F.log_softmax(x, dim=1))
        # return F.log_softmax(x, dim=1),x
