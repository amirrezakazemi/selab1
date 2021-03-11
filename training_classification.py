import numpy as np
import pandas as pd
import sys, os
import torch
import torch.nn as nn
from models import *
from metrics import *
from data_preprocessing import *

# training function at each epoch
def train(model, device, train_loader, optimizer, epoch, loss_fn, dataset):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    total_loss = 0
    batch_idx = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        #loss = loss_fn(output, data.y.view(-1, 1).to(device).float())
        label = data.y.long().to(device)
        if dataset == "davis":
          threshold = 7
        else:
          threshold = 12.1
        label = torch.where(label > threshold, torch.ones(label.shape).to(device), torch.zeros(label.shape).to(device))
        loss = loss_fn(output, label.to(device).long())
        total_loss = loss.item()
        loss.backward()
        optimizer.step()

def predicting(model, device, loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print('Prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
    return total_labels.numpy().flatten(),total_preds.numpy()

