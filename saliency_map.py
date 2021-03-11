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


