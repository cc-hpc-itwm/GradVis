import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
sys.path.append("../toolbox")
import Visualization as vis
import nn_model
import trajectory_plots as tplot
import hessian_functions as hf
import matplotlib.pyplot as plt



"Check else create folders stat_data and data"
if not os.path.exists("stat_data"):
    os.makedirs("stat_data")
if not os.path.exists("data"):
    os.makedirs("data")

###############################################
#########        Define Model       ###########
###############################################



class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = LeNet()
net = net.cuda()

###############################################
#########        Define Dataset     ###########
###############################################

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
"Load CIFAR10 Dataset"
train_data = torch.utils.data.DataLoader(datasets.CIFAR10(root='./cifar_data', train=True, transform=transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomCrop(32, 4),transforms.ToTensor(),normalize,]), download=True),batch_size=256, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.CIFAR10(root='./cifar_data', train=False, transform=transforms.Compose([transforms.ToTensor(),normalize,])),batch_size=128, shuffle=False)

###############################################
#########       Define Training     ###########
###############################################

def train_net(model, epoch):
    model.train()

    lr=0.04
    optimizer = optim.SGD(model.parameters(),lr=lr,momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for batch_id, (data, target) in enumerate(train_data):
        data, target = data.cuda(), target.cuda()
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
    for batch_idx, (inputs, targets) in enumerate(train_data):
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if batch_idx == 30:
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
            data, target = data.cuda(), target.cuda()
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

orig_loss = []
orig_acc = []

torch.save(net.state_dict(), "data/Lenet_minima_point_"+str(0))

for ep in range(1,10+1):
    print("Epoch %i"%(ep))
    tloss = train_net(net,ep)
    orig_loss.append(tloss)
    accu = test(net)
    orig_acc.append(accu)
    torch.save(net.state_dict(), "data/Lenet_minima_point_"+str(ep))

np.save("stat_data/Lenet_orig_loss", orig_loss)
np.save("stat_data/Lenet_orig_acc", orig_acc)

###################################################
######### Visualize Loss and Trajectory ###########
###################################################


num_files = 10
fileindices=np.linspace(0,num_files,num_files + 1)
filesname = ["./data/Lenet_minima_point_"+str(int(i)) for i in fileindices]


nnmodel =  nn_model.PyTorch_NNModel(net,train_vis,filesname[-1])
vis.visualize_eigendirs(nnmodel,filesname,40,"minima_vis_eigen",train_data,nn.CrossEntropyLoss(),proz=.4,percentage=0.01, verbose=True)
tplot.plot_loss_2D("minima_vis_eigen.npz",filename="Lenet_minima_2D_plot_eigen",is_log=True)
tplot.plot_loss_3D("minima_vis_eigen.npz",filename="Lenet_minima_3D_plot_eigen",degrees=50)

###########################################################
######### Calculate Eigenvalue Density Spectrum ###########
###########################################################

hf.stochastic_lanczos(net,train_data,nn.CrossEntropyLoss(),"Lenet_eigenvals",percentage=0.05,num_iters=5,verbose=True)
x1,x2 = hf.get_xlim(["Lenet_eigenvals.npz"])
x = np.linspace(x1-2,x2+2,1000)
func = hf.get_spectrum(x,np.sqrt(1e-2),"Lenet_eigenvalues.npz")

f = plt.figure()
plt.semilogy(x,func)
plt.ylim(1e-9,1e2)
plt.grid(True)
plt.savefig("Lenet_eigenvalues.svg")
