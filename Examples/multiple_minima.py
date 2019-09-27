import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

import Visualization as vis
import nn_model
import trajectory_plots as tplot


"Check else create folders stat_data and data"
if not os.path.exists("stat_data"):
    os.makedirs("stat_data")
if not os.path.exists("data"):
    os.makedirs("data")

###############################################
#########        Define Model       ###########
###############################################


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

"Check whether GPU is available"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = Net()
net = net.to(device)


###############################################
#########        Define Dataset     ###########
###############################################

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

"Load CIFAR10 Dataset"
train_data = torch.utils.data.DataLoader(datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomCrop(32, 4),transforms.ToTensor(),normalize,]), download=True),batch_size=256, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([transforms.ToTensor(),normalize,])),batch_size=128, shuffle=False)



###############################################
#########       Define Training     ###########
###############################################

def train_net(model, epoch):
    model.train()

    lr=0.01
    optimizer = optim.SGD(model.parameters(),lr=lr,momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for batch_id, (data, target) in enumerate(train_data):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output,target)
        loss.backward()
        optimizer.step()

        if batch_id % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_id * len(data), len(train_data.dataset),
                100. * batch_id / len(train_data), loss.item()))

    return loss.item()


###############################################
#########    Define Visualization   ###########
###############################################

def train_vis():
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (inputs, targets) in enumerate(train_data): #One can also replace train_data with test_loader in order to plot the loss landscape of the test set
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx == 20:
            break
    return train_loss/(batch_idx+1)


###############################################
#########        Define Testing     ###########
###############################################

def test(model):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    return 100. * correct / len(test_loader.dataset)



##################################################
######### Train Model and Save Weights ###########
##################################################

"Function in order to reset the weights of the network"
def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

"Function to flatten the network parameters"
def flat_weights_dict(weight_dict):
    vec = []
    for param in weight_dict.values():
        vec.append(param.view(-1))
    return torch.cat(vec)


"Train the network three times to obtain 3 different minima"
for i in range(3):
    net.apply(weight_reset)

    orig_loss = []
    orig_acc = []

    for ep in range(1,20+1):
        print("Epoch %i"%(ep))
        tloss = train_net(net,ep)
        orig_loss.append(tloss)
        accu = test(net)
        orig_acc.append(accu)

    np.save("stat_data/loss_"+str(i),orig_loss)
    np.save("stat_data/accuracy_"+str(i),orig_acc)

    torch.save(net.state_dict(), "data/minima_point_"+str(i))


"Load the minima"
minimum0 = torch.load("data/minima_point_0")
minimum1 = torch.load("data/minima_point_1")
minimum2 = torch.load("data/minima_point_2")

"Flatten their structure"
flat_minimum0 = flat_weights_dict(minimum0)
flat_minimum1 = flat_weights_dict(minimum1)
flat_minimum2 = flat_weights_dict(minimum2)

"Subtract to make two vectors"
v_dir = flat_minimum1-flat_minimum0
w_dir = flat_minimum2-flat_minimum0

###################################################
######### Visualize Loss and Trajectory ###########
###################################################

num_files = 2
fileindices=np.linspace(0,num_files,num_files + 1)
filesname = ["./data/minima_point_"+str(int(i)) for i in fileindices]

nnmodel =  nn_model.PyTorch_NNModel(net, train_vis, filesname[-1])
vis.visualize(nnmodel, filesname, 30, "three_minima_example", v_vec=v_dir.cpu().numpy(), w_vec=w_dir.cpu().numpy(), proz=.4, verbose=True)
tplot.plot_loss_2D("three_minima_example.npz",filename="three_minima_contour",is_log=True)
tplot.plot_loss_3D("three_minima_example.npz",filename="three_minima_3D",height=40,degrees=100,is_log=True)