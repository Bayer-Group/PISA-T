import torch

import os
import numpy as np
from tqdm import tqdm


def get_freer_gpu():
    """
    Helper function that gets the index of the freest GPU available.
    """
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Used > devices")
    memory_used = [int(x.split()[2]) for x in open("devices", "r").readlines()]
    return int(np.argmin(memory_used))


def train_one_epoch(model, data_loader, optimizer, loss_fn, writer, epoch):
    """
    Training Loop.
    """
    model.train()

    device = next(model.parameters()).device.index
    tb_writer = writer

    loop = tqdm(enumerate(data_loader), total=len(data_loader), leave=False)
    loop.set_description(f"Epoch [{epoch + 1}]")
    
    #train_acc = 0
    train_loss = 0
    for _, (X, y) in loop:
        # Forward pass
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss
        # convert dtype for predicting acc.
        #train_acc += acc_fn(torch.sigmoid(pred), y.type(torch.int32))
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    #train_acc /= len(data_loader)

    if tb_writer:
        tb_writer.add_scalar('Loss/train', train_loss, epoch+1)
        #tb_writer.add_scalar('MCC/train', train_acc, epoch+1)
    
    return train_loss#, train_acc


def evaluate(model, data_loader, loss_fn, writer, epoch):
    """
    Evaluation Loop.
    """
    model.eval()

    device = next(model.parameters()).device.index
    tb_writer = writer

    loop = tqdm(enumerate(data_loader), total=len(data_loader), leave=False)
    loop.set_description(f" Validation Step [{epoch + 1}]")

    test_loss = 0
    #test_acc = 0
    with torch.no_grad():
        for _, (X, y) in loop:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y)
            # convert dtype for predicting acc.
            #test_acc += acc_fn(torch.sigmoid(pred), y.type(torch.int32))

        test_loss /= len(data_loader)
        #test_acc /= len(data_loader)
    
    if tb_writer:
        tb_writer.add_scalar('Loss/val', test_loss, epoch+1)
        #tb_writer.add_scalar('MCC/val', test_acc, epoch)

    return test_loss#, test_acc
