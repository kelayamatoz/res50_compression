# -*- coding: utf-8 -*-

'''Deep Compression with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms
import torch.nn.utils.prune as pruner


import os
import argparse

from models import *
from utils import progress_bar

import numpy as np

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Pruning')
parser.add_argument('--loadfile', '-l',
                    default="checkpoint/res18.t7", dest='loadfile')
parser.add_argument('--prune', '-p', default=0.5,
                    dest='prune', help='Parameters to be pruned')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--net', default='res18')
args = parser.parse_args()

prune = float(args.prune)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

# Model
print('==> Building model..')
if args.net == 'res18':
    net = ResNet18()
elif args.net == 'vgg':
    net = VGG('VGG19')
elif args.net == 'res50':
    net = ResNet50()

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


# Load weights from checkpoint.
print('==> Resuming from checkpoint..')
assert os.path.isfile(args.loadfile), 'Error: no checkpoint directory found!'
checkpoint = torch.load(args.loadfile)
net.load_state_dict(checkpoint['net'])

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)

for name, module in net.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        pruner.l1_unstructured(module, name='weight', amount=prune)
    elif isinstance(module, torch.nn.Linear):
        pruner.l1_unstructured(module, name='weight', amount=prune)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # mask pruned weights
        checkpoint['net'] = net.state_dict()

        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # print current sparisty
    for name, module in net.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            print("Sparsity at {} = {}%".format(name, 100. * float(torch.sum(module.weight == 0)) / float(module.weight.numel())))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = net.state_dict()
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/pruned-'+args.net+'-ckpt.t7')
        best_acc = acc


for epoch in range(start_epoch, start_epoch+20):
    train(epoch)
    test(epoch)
    with open("prune-results-"+str(prune)+'-'+str(args.net)+".txt", "a") as f:
        f.write(str(epoch)+"\n")
        f.write(str(best_acc)+"\n")
