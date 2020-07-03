


import torch
import torchvision
import torchvision.transforms as transforms

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data/', train=True,
                                        download=False, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./data/', train=False,
                                       download=False, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                          shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.block1 = nn.Sequential(
                nn.Conv2d(3 , 64 , 3 , padding = 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64 , 64 , 3 , padding = 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size =(2,2),stride = (2,2))
            )
        self.block2 = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding =1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding = 1),
                nn.BatchNorm2d(128), 
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = (2,2), stride = (2,2))
            )
        self.block3 = nn.Sequential(
                nn.Conv2d(128, 256, 3, padding = 1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.Conv2d(256,256,3, padding = 1),
                nn.BatchNorm2d(256) ,
                nn.ReLU(),
                nn.Conv2d(256,256,3, padding = 1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size = (2,2),stride=(2,2))
            )
        self.block4 = nn.Sequential(
                nn.Conv2d(256,512,3, padding = 1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512 , 512 , 3 , padding = 1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512,512,3,padding = 1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2,2), stride = (2,2))
            )
        self.block5 = nn.Sequential(
                nn.Conv2d(512,512,3,padding = 1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512,512,3,padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512,512,3,padding =1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
            )
        self.classifier = nn.Sequential(
                nn.Linear(512,4096),
                nn.ReLU(True),
                nn.Dropout(p=0.65),
                nn.Linear(4096,4096),
                nn.ReLU(True),
                nn.Dropout(p=0.65),
                nn.Linear(4096,10)
            )



    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        #x = F.softmax(x, dim=1)
        return x

net = Net()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
#optimizer = optim.SGD(net.parameters(), lr = 0.001 , momentum = 0.09)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
net = nn.DataParallel(net)

for epoch in range(10):

    running_loss = 0.
    batch_size = 100
    ratio=1
    for i, data in enumerate(
        torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                        shuffle=True, num_workers=2), 0):
        
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        
        
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
       
        get the top k biggest gradient
        for ind, p in enumerate(net.parameters()):
            if len(p.data.size()) != 1:
                grad = p.grad.view(-1).to(device)
                grad_pos = (((grad>0).float())*grad).abs()
                grad_neg = (((grad<0).float())*grad).abs()
                grad_size = grad.shape 
                grad_zero_pos = torch.zeros(grad.shape).to(device)
                grad_zero_neg = torch.zeros(grad.shape).to(device)
                grad_sum = torch.zeros(grad.shape).to(device)
                top_grad_size = int(grad_size[0] * ratio)
                topk_posvalue , topk_posind = torch.topk(grad_pos, top_grad_size)
                topk_negvalue , topk_negind = torch.topk(grad_neg, top_grad_size)
                grad_zero_pos[topk_posind] = grad_pos[topk_posind] 
                grad_zero_neg[topk_negind] = grad_neg[topk_negind]
                grad_sum = grad_zero_pos + (-1*grad_zero_neg)
                shape = p.grad.shape
                tmp = grad_sum.reshape(shape)
                p.grad = tmp.detach()


        #tale the top k biggest weight
        #for ind, p in enumerate(net.parameters()):
        #    if len(p.data.size()) != 1:
        #        grad = p.grad.view(-1).to(device)
        #        weight = p.data.view(-1).to(device)
        #        #print("grad shaoe = ",grad.shape)
        #        #print("weight shape = ",weight.shape)
        #        weight_pos = (((weight>0).float())*weight).abs()
        #        weight_neg = (((weight<0).float())*weight).abs()
        #        weight_size = weight.shape
        #        grad_pos = (((weight>0)).float()*grad).abs()
        #        grad_neg = (((weight<0)).float()*grad).abs()
        #        weight_zero_pos = torch.zeros(weight.shape).to(device)
        #        weight_zero_neg = torch.zeros(weight.shape).to(device)
        #        weight_sum = torch.zeros(weight.shape).to(device)
        #        grad_zero_pos = torch.zeros(grad.shape).to(device)
        #        grad_zero_neg = torch.zeros(grad.shape).to(device)
        #        grad_sum = torch.zeros(weight.shape).to(device)
        #        top_weight_size = int(weight_size[0]*ratio)
        #        topk_posvalue , topk_posind = torch.topk(weight_pos, top_weight_size)
        #        topk_negvalue , topk_negind = torch.topk(weight_neg, top_weight_size)
        #        grad_zero_pos[topk_posind] = grad_pos[topk_posind]
        #        grad_zero_neg[topk_negind] = grad_neg[topk_negind]
        #        grad_sum = grad_zero_pos + (-1*grad_zero_neg)
        #        shape = p.grad.shape
        #        tmp = grad_sum.reshape(shape)
        #        p.grad = tmp.detach()
               


        optimizer.step()
        
        print('[%d, %5d] loss: %.4f' %(epoch + 1, (i+1)*batch_size, loss.item()))

print('Finished Training')

torch.save(net, './nn_weight/cifar10.pkl')
# net = torch.load('cifar10.pkl')

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))








