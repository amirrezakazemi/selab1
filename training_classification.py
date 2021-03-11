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


def run(dataset_index, resampled_flag):
    pretrained_model = GINConvNet()
    pretrained_model.load_state_dict(torch.load('model_davis_resampled.model'))
    pretrained_model_dict = pretrained_model.state_dict()

    new_model = GINConvNetClassification()
    new_model_dict = new_model.state_dict()
    pretrained_model_dict = {k: v for k, v in pretrained_model_dict.items() if k in new_model_dict}
    new_model_dict.update(pretrained_model_dict)
    new_model.load_state_dict(new_model_dict)
    model = new_model
    for name, param in model.named_parameters():
        if "classifyout" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    datasets = [['davis', 'kiba'][dataset_index]]
    model_st = "GINConvNetClassification"
    dataset = datasets[0]
    TRAIN_BATCH_SIZE = 512
    TEST_BATCH_SIZE = 512
    LR = 0.00005
    NUM_EPOCHS = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_file_name = 'model_' + dataset + 'classificatio.model'
    result_file_name = 'result_' + dataset + 'classification.csv'
    print('\nRunning on ' + dataset)
    print('Learning rate: ', LR)
    print('EpochsNumber: ', NUM_EPOCHS)

    # Main program: iterate over different datasets
    for dataset in datasets:
        print('\nrunning on ' + dataset)
        processed_data_file_train = 'data/processed/' + dataset + '_train.pt'
        processed_data_file_test = 'data/processed/' + dataset + '_test.pt'
        if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
            print('please run create_data.py to prepare data in pytorch format!')
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

            model = model.to(device)
            loss_fn = nn.CrossEntropyLoss(weight=torch.Tensor([1., 2.]).to(device))
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
            best_mse = 1000
            best_ci = 0
            best_epoch = -1
            best_model = None
            # for validation
            best_f1 = 0
            best_recall = 0
            best_acc = 0
            best_prec = 0

            for epoch in range(1, NUM_EPOCHS + 1):
                print('\nEpoch %d:' % epoch)
                train(model, device, train_loader, optimizer, epoch + 1, loss_fn, dataset)
                G, P = predicting(model, device, val_loader)
                if dataset == "davis":
                    one_hot_G, one_hot_P = np.where(G > 7, 1, 0), np.argmax(P, axis=1)
                else:
                    one_hot_G, one_hot_P = np.where(G > 12.1, 1, 0), np.argmax(P, axis=1)
                print(np.count_nonzero(one_hot_G == 1), np.count_nonzero(one_hot_P == 1))
                print("accuracy", accuracy(one_hot_G, one_hot_P))
                print("recall", recall(one_hot_G, one_hot_P))
                print("precision", precision(one_hot_G, one_hot_P))
                print("f1", f1(one_hot_G, one_hot_P))
                if f1(one_hot_G, one_hot_P) > best_f1:
                    best_f1 = f1(one_hot_G, one_hot_P)
                    best_model = model
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
            if dataset == "davis":
                one_hot_G, one_hot_P = np.where(G > 7, 1, 0), np.argmax(P, axis=1)
            else:
                one_hot_G, one_hot_P = np.where(G > 12.1, 1, 0), np.argmax(P, axis=1)
            print("test_accuracy: ", accuracy(one_hot_G, one_hot_P))
            print("test_recall", recall(one_hot_G, one_hot_P))
            print("test_precision", precision(one_hot_G, one_hot_P))
            print("test_f1", f1(one_hot_G, one_hot_P))

    torch.save(model.state_dict(), 'model_davis_classification.pt')

