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


# run a model on specific dataset
def run(dataset_index, resampled_flag):
    datasets = [['davis', 'kiba'][dataset_index]]
    modeling = [GINConvNet][0]
    model_st = modeling.__name__
    dataset = datasets[0]
    model = None
    TRAIN_BATCH_SIZE = 512
    TEST_BATCH_SIZE = 64
    LR = 0.0005
    NUM_EPOCHS = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_file_name = 'model_' + dataset + str(resampled_flag) + '.model'
    result_file_name = 'result_' + dataset + str(resampled_flag) + '.csv'
    print('\nRunning on ' + dataset)
    print('Learning rate: ', LR)
    print('EpochsNumber: ', NUM_EPOCHS)

    processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        print('run data_preprocessing.py to prepare data in .pt format!')
    else:
        if resampled_flag == 1:
            train_data = TestbedDataset(root='data', dataset=dataset + '_resampled_train')
        else:
            train_data = TestbedDataset(root='data', dataset=dataset + '_train')

        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

        val_data = TestbedDataset(root='data', dataset=dataset + '_val')
        val_loader = DataLoader(val_data, batch_size=TEST_BATCH_SIZE, shuffle=True)

        test_data = TestbedDataset(root='data', dataset=dataset + '_test')
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)

        model = modeling().to(device)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
        best_mse = 10000
        best_ci = 0
        best_epoch = -1
        best_model = None
        ##for validation
        best_f1 = 0
        best_recall = 0
        best_acc = 0
        best_prec = 0

        for epoch in range(1, NUM_EPOCHS + 1):
            print('\nEpoch %d:' % epoch)
            train(model, device, train_loader, optimizer, epoch, loss_fn)
            G, P = predicting(model, device, val_loader)
            ret = [mse(G, P), ci(G, P)]
            print("loss: ", ret[0])
            print("ci", ret[1])
            if ret[0] < best_mse:
                torch.save(model.state_dict(), model_file_name)
                best_model = model
                with open(result_file_name, 'w') as f:
                    f.write(','.join(map(str, ret)))
                best_epoch = epoch
                best_mse = ret[0]
                best_ci = ret[1]
                print('Improvement at epoch ', best_epoch, '; best_mse,best_ci:', best_mse, best_ci)
            else:
                print('No improvement since epoch ', best_epoch, '; best_mse,best_ci:', best_mse, best_ci)
            if dataset == "davis":
                one_hot_G, one_hot_P = np.where(G > 7, 1, 0), np.where(P > 7, 1, 0)
            else:
                one_hot_G, one_hot_P = np.where(G > 12.1, 1, 0), np.where(P > 12.1, 1, 0)
            print("validation_accuracy: ", accuracy(one_hot_G, one_hot_P))
            print("validation_recall", recall(one_hot_G, one_hot_P))
            print("validation_precision", precision(one_hot_G, one_hot_P))
            print("validation_f1", f1(one_hot_G, one_hot_P))

            if f1(one_hot_G, one_hot_P) > best_f1:
                best_f1 = f1(one_hot_G, one_hot_P)
            if precision(one_hot_G, one_hot_P) > best_prec:
                best_prec = precision(one_hot_G, one_hot_P)
            if recall(one_hot_G, one_hot_P) > best_recall:
                best_recall = recall(one_hot_G, one_hot_P)
            if accuracy(one_hot_G, one_hot_P) > best_acc:
                best_acc = accuracy(one_hot_G, one_hot_P)
        model = best_model
        print("best validation accuracy", best_acc)
        print("best validation recall", best_recall)
        print("best validation precision", best_prec)
        print("best validation f1", best_f1)
        G, P = predicting(model, device, test_loader)
        ret = [mse(G, P), ci(G, P)]
        print("loss: ", ret[0])
        print("ci", ret[1])
        if dataset == "davis":
            one_hot_G, one_hot_P = np.where(G > 7, 1, 0), np.where(P > 7, 1, 0)
        else:
            one_hot_G, one_hot_P = np.where(G > 12.1, 1, 0), np.where(P > 12.1, 1, 0)
        print("test_accuracy: ", accuracy(one_hot_G, one_hot_P))
        print("test_recall", recall(one_hot_G, one_hot_P))
        print("test_precision", precision(one_hot_G, one_hot_P))
        print("test_f1", f1(one_hot_G, one_hot_P))

# run(int(sys.argv[1]), int(sys.argv[2]))