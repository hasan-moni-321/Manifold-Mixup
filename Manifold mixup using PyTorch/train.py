import numpy as np
import pandas as pd 

import gc

from sklearn.model_selection import train_test_split

import torch 
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim import lr_scheduler
import torch.optim as optim


import dataset_file
import model 
import engine
import train 


# Some Global variable
BATCH_SIZE = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Reading dataset 
df_train = pd.read_csv('/home/hasan/Data Set/Kannada-MNIST/train.csv')
target = df_train['label']
df_train.drop('label', axis=1, inplace=True)

X_test = pd.read_csv('/home/hasan/Data Set/Kannada-MNIST/test.csv')
X_test.drop('id', axis=1, inplace=True)



# dividing dataset into train and validation
X_train, X_dev, y_train, y_dev = train_test_split(df_train, target, stratify=target, random_state=42, test_size=0.2)
print(f"Shape of X_train is {X_train.shape} Shape of X_dev is {X_dev.shape}")
print(f"Shape of y_train is {y_train.shape} Shape of y_dev is {y_dev.shape}")




# Put some augmentation on training data
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
    transforms.ToTensor()
])

# Test data without augmentation
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])



# Calling Custom Dataset
train_dataset = dataset_file.CharData(X_train, y_train, train_transform)
dev_dataset = dataset_file.CharData(X_dev, y_dev, test_transform)
test_dataset = dataset_file.CharData(X_test, transform=test_transform)



# Create data generators - they will produce batches
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
dev_loader = DataLoader(dataset=dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)



# Model, Device, Optimizer and Learning rate scheduler 
model = model.Net(10)
model = model.to(device)

n_epochs = 3

optimizer = optim.Adam(model.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=n_epochs // 4, gamma=0.1)




# Training the Model 
history = pd.DataFrame()

for epoch in range(n_epochs):
    #torch.cuda.empty_cache()
    gc.collect()
    # train_loader, model, device, optimizer, epoch, history=None
    engine.train(train_loader, model, device, optimizer, epoch, exp_lr_scheduler, history)
    # dev_loader, model, device, epoch, history=None
    engine.evaluate(dev_loader, model, device, optimizer, epoch, history)

