'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage

import os
import argparse
import time
from models import *
from utils import progress_bar

import glob
import cv2
import numpy as np
from torch.autograd import Variable

show = ToPILImage()
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

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

transform_test2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


# Training
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
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch, saveFlag=True):
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

    # Save checkpoint.
    if(saveFlag):
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            cur_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            file_name = "./checkpoint/{}-acc{}-ckpt.pth".format(cur_time, acc)
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/ckpt.pth')
            torch.save(state, file_name)
            best_acc = acc


def testOneImage():
    # (data, label) = testset[100]
    # print(classes[label])
    # # (data + 1) / 2是为了还原被归一化的数据
    # show(data / 2 - 0.5).resize((100, 100)).show()

    image_num = 4

    dataiter = iter(testloader)
    images, labels = dataiter.next() # 返回4张图片及标签
    print(' '.join('%11s'%classes[labels[j]] for j in range(image_num)))
    # show(torchvision.utils.make_grid(images[:image_num] / 2 - 0.5)).resize((400, 100)).show()

    with torch.no_grad():
        outputs = net(images[:image_num])
        _, predicted = outputs.max(1)
        print('预测结果: ', ' '.join('%5s' % classes[predicted[j]] for j in range(image_num)))


def testFolderImages():
    for jpgfile in glob.glob(r'./cifar10_val/*.jpeg'):
        print(jpgfile)  # 打印图片名称，以与结果进行对照
        img = cv2.imread(jpgfile)  # 读取要预测的图片，读入的格式为BGR
        image = cv2.resize(img, (32, 32))
        # cv读入数据是32x32x3
        # cv2.imshow("f", image)
        # cv2.waitKey(0)
        image = np.expand_dims(image, 0).astype(np.float32)
        tensor_image = torch.from_numpy(image)
        # torch.transpose(tensor_image, 2, 3)
        tensor_image = tensor_image.transpose(2, 3).contiguous()
        tensor_image = tensor_image.transpose(1, 2).contiguous()
        tensor_image = tensor_image.to(device)
        torch.set_default_tensor_type(torch.DoubleTensor)
        # 模型要求的输入tensor size是(1,3,32,32)

        output = net(Variable(tensor_image))
        _, predicted = output.max(1)
        print("pp", classes[predicted])


test_mode = False
# testOneImage()
#testFolderImages()
# x = torch.randn(8, 3, 5, 4)
# y = x.transpose(2, 3)  # 交换第二与第三维度
# print(y.shape)
# cur_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
# file_name = "./checkpoint/{}-acc{}-ckpt.pth".format(cur_time, 80)
# print(file_name)

# dataiter = iter(trainloader)
# images, labels = dataiter.next() # 返回4张图片及标签
# print(' '.join('%11s'%classes[labels[j]] for j in range(4)))
# show(torchvision.utils.make_grid(images / 2 - 0.5)).resize((400,100)).show()

if test_mode:
    test(1, saveFlag=False)
else:
    train_epoch = 1  # 1个epoch要训练40分钟

    for epoch in range(start_epoch, start_epoch + train_epoch):
        train(epoch)
        test(epoch)
        scheduler.step()
