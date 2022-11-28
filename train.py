"""
Training : Train model on Cityscapes dataset
"""
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from segformer import segformer_mit_b3
import segmentation_models_pytorch as smp

import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import meanIoU                  # metric class
from utils import plot_training_results

# utility functions to get dataset and dataloaders
from utils import get_BDD_datasets



def evaluate_model(model, dataloader, criterion, metric_class, num_classes, device):
    '''evaluate model on dataset'''
    model.eval()
    total_loss = 0.0
    metric_object = metric_class(num_classes)

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, total=len(dataloader)):
            inputs = inputs.to(device)
            labels = labels.to(device)                
            y_preds = model(inputs)

            # calculate loss
            loss = criterion(y_preds, labels)
            total_loss += loss.item()

            # update batch metric information            
            metric_object.update(y_preds.cpu().detach(), labels.cpu().detach())

    evaluation_loss = total_loss / len(dataloader)
    evaluation_metric = metric_object.compute()
    return evaluation_loss, evaluation_metric


def train_validate_model(model, num_epochs, model_name, criterion, optimizer, 
                         device, dataloader_train, dataloader_valid, 
                         metric_class, metric_name, num_classes, lr_scheduler = None,
                         output_path = '.'):
    '''training process'''
    # initialize placeholders for running values
    results = []    
    min_val_loss = np.Inf
    len_train_loader = len(dataloader_train)

    # move model to device
    model.to(device)
    
    for epoch in range(num_epochs):
        print(f"Starting {epoch + 1} epoch ...")
        
        # Training
        model.train()
        train_loss = 0.0
        for inputs, labels in tqdm(dataloader_train, total=len_train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device) 

            # Forward pass
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)
            train_loss += loss.item()
              
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # adjust learning rate
            if lr_scheduler is not None:
                lr_scheduler.step()
            
        # compute per batch losses, metric value
        train_loss = train_loss / len(dataloader_train)
        validation_loss, validation_metric = evaluate_model(
                        model, dataloader_valid, criterion, metric_class, num_classes, device)

        print(f'Epoch: {epoch+1}, trainLoss:{train_loss:6.5f}, validationLoss:{validation_loss:6.5f}, {metric_name}:{validation_metric: 4.2f}')
        
        # store results
        results.append({'epoch': epoch, 
                        'trainLoss': train_loss, 
                        'validationLoss': validation_loss, 
                        f'{metric_name}': validation_metric})
        
        # if validation loss has decreased, save model and reset variable
        if validation_loss <= min_val_loss:
            min_val_loss = validation_loss
            torch.save(model.state_dict(), f"{output_path}/{model_name}.pt")
            # torch.jit.save(torch.jit.script(model), f"{output_path}/{model_name}.pt")


    # plot results
    results = pd.DataFrame(results)
    plot_training_results(results, model_name)
    return results

if __name__ == '__main__':
    """BDD10k / 100K Dataset"""
    imgs = 'bdd10k_images'             # bdd100k_images
    task = 'bdd10k_sem_seg_labels'     # bdd100k_drivable_labels
    train_set, val_set = get_BDD_datasets(rootDir='dataset', imgs=imgs, task=task)
    sample_image, sample_label = train_set[0]
    print(f"There are {len(train_set)} train images, {len(val_set)} validation images")
    print(f"Input shape = {sample_image.shape}, output label shape = {sample_label.shape}")
 
    train_dataloader = DataLoader(train_set, batch_size=2, drop_last=True)
    val_dataloader   = DataLoader(val_set, batch_size=2)

    """Model Training"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

    if task == 'bdd100k_drivable_labels':
        # MODEL HYPERPARAMETERS
        N_EPOCHS = 7
        NUM_CLASSES = 3
        MAX_LR = 3e-4
        MODEL_NAME = f'segformer_mit_b3_bdd_3CLS_CEloss'
        # loss function
        # criterion = smp.losses.DiceLoss('multiclass', classes=[0,1,2], log_loss = True, smooth=1.0)
        criterion = nn.CrossEntropyLoss()
    elif task == 'bdd10k_sem_seg_labels':
        # MODEL HYPERPARAMETERS
        N_EPOCHS = 30
        NUM_CLASSES = 19
        MAX_LR = 1e-3
        MODEL_NAME = 'segformer_mit_b3_bdd_19CLS_CE_loss'
        # loss function
        # criterion = nn.CrossEntropyLoss(ignore_index=19)
        criterion = smp.losses.FocalLoss('multiclass', ignore_index=19)

    # create model, load imagenet pretrained weights
    model = segformer_mit_b3(in_channels=3, num_classes=NUM_CLASSES).to(device)
    model.backbone.load_state_dict(torch.load('segformer_mit_b3_imagenet_weights.pt', map_location=device))

    # create optimizer, lr_scheduler and pass to training function
    optimizer = optim.Adam(model.parameters(), lr=MAX_LR)
    scheduler = OneCycleLR(optimizer, max_lr= MAX_LR, epochs = N_EPOCHS, 
                        steps_per_epoch = len(train_dataloader), div_factor=10)


    train_validate_model(model, N_EPOCHS, MODEL_NAME, criterion, optimizer, 
                        device, train_dataloader, val_dataloader, meanIoU, 'meanIoU',
                        NUM_CLASSES, lr_scheduler = scheduler, output_path = "backup")