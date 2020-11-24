import numpy as np 

import torch 
import torch.nn as nn
import torch.nn.functional as F



def criterion(input, target, size_average=True):
    """Categorical cross-entropy with logits input and one-hot target"""
    l = -(target * torch.log(F.softmax(input, dim=1) + 1e-10)).sum(1)
    if size_average:
        l = l.mean()
    else:
        l = l.sum()
    return l




def train(train_loader, model, device, optimizer, epoch, exp_lr_scheduler, history=None):
    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        
        # mixup
        alpha = 2
        lam = np.random.beta(alpha, alpha)
        shuffle = torch.randperm(data.shape[0])
        target = lam * target + (1 - lam) * target[shuffle]
        
        optimizer.zero_grad()
        output = model([data, shuffle, lam])
        loss = criterion(output, target)
        if history is not None:
            history.loc[epoch + batch_idx / len(train_loader), 'train_loss'] = loss.data.cpu().numpy()
        
        loss.backward()
        optimizer.step()
        
        if (batch_idx + 1) % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLR: {:.6f}\tLoss: {:.6f}'.format(
                epoch, (batch_idx + 1) * len(data), len(train_loader.dataset),
                100. * (batch_idx + 1) / len(train_loader),
                optimizer.state_dict()['param_groups'][0]['lr'],
                loss.data))
    exp_lr_scheduler.step()



def evaluate(dev_loader, model, device, epoch, history=None):
    model.eval()
    loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in dev_loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)

            loss += criterion(output, target, size_average=False).data

            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.max(1, keepdim=True)[1].data.view_as(pred)).cpu().sum().numpy()
    
    loss /= len(dev_loader.dataset)
    accuracy = correct / len(dev_loader.dataset)
    
    if history is not None:
        history.loc[epoch, 'dev_loss'] = loss.cpu().numpy()
        history.loc[epoch, 'dev_accuracy'] = accuracy
    
    print('Dev loss: {:.4f}, Dev accuracy: {}/{} ({:.3f}%)\n'.format(
        loss, correct, len(dev_loader.dataset),
        100. * accuracy))


