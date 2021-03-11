import torch
import torch.nn as nn
from models import *
from metrics import *
import torch.nn.functional as F
from data_preprocessing import *
from training_classification import predicting
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib


def generate_bp_saliency(model, input, target):
    input.x.requires_grad = True
    input.requires_grad = True
    model.zero_grad()
    output = model(input)
    model.grad_x.retain_grad()
    model.grad_embedded_xt.retain_grad()
    grad_outputs = torch.zeros_like(output)
    grad_outputs[:, target] = 1
    output.backward(gradient=grad_outputs, retain_graph=True)

    return model.grad_x.grad, model.grad_embedded_xt.grad


def generate_gc_saliency(model, input, target):
    input.x.requires_grad = True
    model.zero_grad()
    output = model(input)
    model.grad_x.retain_grad()
    model.grad_embedded_xt.retain_grad()
    model.grad_conv_xt.retain_grad()
    model.grad_conv.retain_grad()
    # print(target)
    grad_outputs = torch.zeros_like(output)
    grad_outputs[:, target] = 1
    # gradient = torch.autograd.grad(outputs=output, inputs = , grad_outputs = grad_outputs, retain_graph=True, allow_unused=True)
    output.backward(gradient=grad_outputs, retain_graph=True)
    return torch.sum(model.grad_conv.grad) * model.conv, torch.sum(model.grad_conv_xt.grad) * model.conv_xt


def guided_relu_hook(module, grad_in, grad_out):
    return (torch.clamp(grad_in[0], min=0.0),)


def generate_guided_bp_saliency(model, input, target):
    input.x.requires_grad = True
    input.requires_grad = True
    model.zero_grad()
    for module in model.modules():
        if type(module) == nn.ReLU:
            module.register_backward_hook(guided_relu_hook)
    output = model(input)
    model.grad_x.retain_grad()
    model.grad_embedded_xt.retain_grad()
    grad_outputs = torch.zeros_like(output)
    grad_outputs[:, target] = 1
    output.backward(gradient=grad_outputs, retain_graph=True)

    return model.grad_x.grad, model.grad_embedded_xt.grad


def generate_guided_gc_saliency(model, input, target):
    a, b = generate_gc_saliency(model, input, target)
    c, d = generate_guided_bp_saliency(model, input, target)

    target_gc = F.pad(b[0], [0, 7, 0, 1000 - 32], "constant", 1)
    drug_gc = F.pad(a, [0, 78 - 32, 0, 0])

    drug_gc, target_gc = F.relu(drug_gc), F.relu(target_gc)

    drug_bp, target_bp = c, d[0]

    drug_ggc = torch.mul(drug_gc, drug_bp)

    target_ggc = torch.mul(target_gc, target_bp)

    return drug_ggc, torch.unsqueeze(target_ggc, 0)


dataset = "davis"
val_data = TestbedDataset(root='data', dataset=dataset + '_test')
TEST_BATCH_SIZE = 512
val_loader = DataLoader(val_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
model = GINConvNetClassification()
model.load_state_dict(torch.load('model_davis_classification.pt'))
for name, param in model.named_parameters():
    param.requires_grad = False

cuda_name = "cuda:0"
device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
G, P = predicting(model, device, val_loader)
one_hot_G, one_hot_P = np.where(G > 7, 1, 0), np.argmax(P, axis=1)
jam = one_hot_G + one_hot_P
index = (jam == 2).nonzero()

print("number of true-identified pairs", index[0].shape)

num_selected_words = {}
methods = [generate_bp_saliency, generate_guided_bp_saliency, generate_guided_gc_saliency]
for method in methods:
    num_selected_words[method.__name__] = list()
    cnt = 0
    for i in index[0]:
        input = val_data[int(i)]
        input = input.to(device)
        a, b = method(model, input, 1)
        drug = torch.sum(abs(a), 1)
        target = torch.sum(abs(b[0]), 1)
        average_drug, average_target = torch.mean(drug), torch.mean(target)
        std_drug, std_target = torch.std(drug), torch.std(target)
        imp_drug = (drug > average_drug + std_drug).nonzero()
        imp_target = (target > average_target + 2 * std_target).nonzero()
        num_selected_words[method.__name__].append(imp_target.shape[0] + imp_drug.shape[0])
        # input.target[0, imp_target] = 0
        input.x[imp_drug, :] = 0
        input.x = Variable(input.x, requires_grad=True)
        input.requires_grad = True
        model.zero_grad()
        out = model(input)
        out = np.argmax(out.cpu().detach().numpy())
        if out == 0:
            cnt += 1

    print("number of changed outcomes by altering drug " + method.__name__, cnt)
    fig = plt.figure()
    plt.hist(np.array(num_selected_words[method.__name__]), 5)
    fig.savefig('hist_' + method.__name__ + '.png')











