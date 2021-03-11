import numpy as np
import pandas as pd
import sys, os
import torch
import torch.nn as nn
from models import *
from metrics import *
from data_preprocessing import *
from sklearn.utils import resample


# training function at each epoch
def train(model, device, train_loader, optimizer, epoch, loss_fn):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()