# -*- coding: utf-8 -*-
"""
VGG16 Implementation in PyTorch

@author: Suman Bhurtel
"""

# Import libraries
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision import transforms 
import torch.optim as optim


# Data Augmentation
RC = transforms.RandomCrop((32, 32), padding=4)
RHF = transforms.RandomHorizontalFlip()
RShift = transforms.ColorJitter(brightness=12, contrast=5, saturation=3, hue=0)
RR = transforms.RandomRotation(degrees=(15, 45), fill=(0))
NRM = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
TT = transforms.ToTensor()
RS = transforms.Resize(224)


# Transforms object for trainset with augmentation
tranform_train_with_aug = transforms.Compose([RC,RShift,RR, RHF, RS, TT, NRM]) # with all augmentation (Experiment 2)
tranform_train_with_no_aug = transforms.Compose([RS, TT, NRM]) # with no augmentation (Experiment 1)
tranform_train_with_RC = transforms.Compose([RC,RS, TT, NRM]) # with random crop augmentation (Experiment 3)
tranform_train_with_RHF = transforms.Compose([RHF, RS, TT, NRM]) # with random Horizontal flip (Experiment 4)
tranform_train_with_Shift = transforms.Compose([RShift, RS, TT, NRM]) # with random Shift (Experiment 5)
tranform_train_with_Rot = transforms.Compose([RR, RS, TT, NRM]) # with random Rotation (Experiment 6)

# Transforms object for testset with NO augmentation
tranform_test_without_aug = transforms.Compose([RS, TT, NRM])



#preparing the train, validation and test dataset
torch.manual_seed(43)
train_ds = CIFAR10("data/", train=True, download=True, transform=tranform_train_with_Rot) # Use augmented transformation here in train data
val_size = 10000
train_size = len(train_ds) - val_size
train_ds, val_ds = random_split(train_ds, [train_size, val_size])
test_ds = CIFAR10("data/", train=False, download=True, transform=tranform_test_without_aug)

#passing the train, val and test datasets to the dataloader
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)
test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)


# Building the  vgg_16 model
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(25088, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.maxpool(x)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.maxpool(x)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        # reduce overfitting
        x = F.dropout(x, 0.5) 
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.5)
        x = self.fc3(x)
        return x
    
# Train model preparation in gpu or cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

model = VGG16()
model = model.to(device=device) 

## Loss and optimizer
learning_rate = 1e-4 
load_model = True
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr= learning_rate) 
loss_all_train=[] # empty list for all train loss
loss_all_val=[] # empty list for all validation loss

acc_all_train = [] # empty list for all train accuracy
acc_all_val = [] # empty list for all validation accuracy

#Training
for epoch in range(30):
    loss_ep = 0
    loss_ep_val = 0
    num_correct_train = 0
    num_samples_train = 0
    
    for batch_idx, (data, targets) in enumerate(train_dl):
        data = data.to(device=device)
        targets = targets.to(device=device)
        optimizer.zero_grad()
        scores = model(data)
        loss = criterion(scores,targets)
        loss.backward()
        optimizer.step()
        loss_ep += loss.item()
        _, predictions_train = scores.max(1)
        num_correct_train += (predictions_train == targets).sum()
        num_samples_train += predictions_train.size(0)
    print(
        f"Got {num_correct_train} / {num_samples_train} with accuracy {float(num_correct_train) / float(num_samples_train) * 100:.2f}"
    )

    print(f"Loss in epoch {epoch} :::: {loss_ep/len(train_dl)}")
    loss_all_train.append(loss_ep/len(train_dl))
    acc_all_train.append(float(num_correct_train)/float(num_samples_train)*100)

    with torch.no_grad():
        num_correct = 0
        num_samples = 0
        for batch_idx, (data,targets) in enumerate(val_dl):
            data = data.to(device=device)
            targets = targets.to(device=device)
            scores = model(data)
            _, predictions = scores.max(1)
            num_correct += (predictions == targets).sum()
            num_samples += predictions.size(0)
            loss_val = criterion(scores,targets)
            loss_ep_val += loss_val.item()
        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
        )
        print(f"Validation Loss in epoch {epoch} :::: {loss_ep_val/len(val_dl)}")
        loss_all_val.append(loss_ep_val/len(val_dl))
        acc_all_val.append(float(num_correct)/float(num_samples)*100)


#print("The total train loss: ",loss_all_train)
#print ("the total validation loss: ",loss_all_val)     


  
# Save Model
torch.save(model.state_dict(), "vgg16_cifar.pt") 
model = VGG16()
model.load_state_dict(torch.load("vgg16_cifar.pt"))
model.eval()


#Testing Model
num_correct = 0
num_samples = 0
for batch_idx, (data,targets) in enumerate(test_dl):
    data = data.to(device="cpu")
    targets = targets.to(device="cpu")
    ## Forward Pass
    scores = model(data)
    _, predictions = scores.max(1)
    num_correct += (predictions == targets).sum()
    num_samples += predictions.size(0)
print(
    f"Got {num_correct} / {num_samples} with Test accuracy {float(num_correct) / float(num_samples) * 100:.2f}"
)


# ploting for train and validation accuracy
def plot_Acc_Res(x,y):
    plt.figure()
    line1, = plt.plot(x)
    line2, = plt.plot(y)
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.legend((line1, line2), ("train Accuracy"," Validation Accuracy"))
    plt.ylim([10, 100])
    plt.show()


def plot_loss_Res(x,y):
    plt.figure()
    line1, = plt.plot(x)
    line2, = plt.plot(y)
    plt.xlabel("Epochs")
    plt.ylabel("loss")
    plt.grid()
    plt.legend((line1, line2), ("train loss"," Validation loss"))
    plt.ylim([-1, 3])
    plt.show()


plot_loss_Res(loss_all_train, loss_all_val)
plot_Acc_Res(acc_all_train, acc_all_val)
