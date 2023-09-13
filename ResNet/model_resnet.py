import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torchvision.models import  resnet50,  resnet34#, ResNet50_Weights, ResNet34_Weights,
import torchvision.models
from typing import Tuple


def initialize_model(lr: float, model_name : str, num_classes : int, 
                     feature_extract : bool
                     ) -> Tuple[torchvision.models.ResNet, torch.optim.AdamW, 
                                torch.nn.modules.loss.CrossEntropyLoss, torch.optim.lr_scheduler.CosineAnnealingLR]:
    """
    Initializes a pretrained model, adds a linear layer for predicting the specified num_classes.
    Also sends the model to the device.
    Defines the learning optimizer and criterion. Returns the initialized model, criterion, and optimizer.

    Parameters
    -----------
        model_name : str
            Name of the model that needs to be initialized
        num_classes : int
            Number of classes the model needs to be able to classify
        feature_extract : bool
            Boolean stating if the base of the model needs to be frozen

    Returns
    -----------
        model : torchvision.models.ResNet
            ResNet initialized model with an added linear layer for predicting the specified num_classes
        optimizer : torch.optim.Adam
            Adam optimizer
        criterion : torch.nn.modules.loss.CrossEntropyLoss
            Loss function for learning criterion: cross entropy loss
    """
    
    model = None

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    if model_name == 'resnet50':
        """
        ResNet50 model
        """
        model = resnet50(pretrained=True).to(device=device)
        
        for param in model.parameters():
            param.requires_grad = feature_extract   
        
        # Add last linear layer for prediction of num_classes
        model.fc = nn.Sequential(
                    # nn.Linear(2048, 128),
                    # nn.ReLU(inplace=True),
                    nn.Linear(2048, num_classes)).to(device=device)
    elif model_name == 'resnet34':
        """
        ResNet34 model
        """
        model = resnet34(pretrained=True).to(device=device)
        
        for param in model.parameters():
            param.requires_grad = feature_extract   
        
        # Add last linear layer for prediction of num_classes
        model.fc = nn.Sequential(
                    # nn.Linear(2048, 128),
                    # nn.ReLU(inplace=True),
                    nn.Linear(512, num_classes)).to(device=device)
    else:
        print('Invalid model name. Pick from the following: resnet or resnet34 ')

    torch.nn.DataParallel(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.fc.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,T_max=425)

    return model, optimizer, criterion, scheduler

def load_model(model: str, device : torch.device, path : str, num_classes : int) -> torchvision.models.ResNet:
    """
    Loads a trained model and sends it to the device. Returns the loaded model.

    Parameters
    -----------
        device : torch.device
            Device to which the model is send
        path : str
            Destination path of where the model weights are stored
        num_classes : int
            Number of classes the model needs to be able to classify
    """
    if model == 'resnet50':
        loaded_model = resnet50(pretrained=True).to(device=device)
        loaded_model.fc = nn.Sequential(
                    # nn.Linear(2048, 128),
                    # nn.ReLU(inplace=True),
                    nn.Linear(2048, num_classes)).to(device=device)
        
        loaded_model.load_state_dict(torch.load(path))
    if model == 'resnet34':
        loaded_model = resnet34(pretrained=True).to(device=device)
        loaded_model.fc = nn.Sequential(
                    # nn.Linear(2048, 128),
                    # nn.ReLU(inplace=True),
                    nn.Linear(512, num_classes)).to(device=device)
        
        loaded_model.load_state_dict(torch.load(path))
    return loaded_model


def save_ckp(state,  checkpoint_dir):
    epoch_n = state['epoch']
    f_path = checkpoint_dir + 'checkpoint_{0}.pt'.format(epoch_n)
    torch.save(state, f_path)


def load_ckp(checkpoint_path : str, model:torchvision.models.ResNet,
             optimizer :torch.optim.AdamW, lr_scheduler : torch.optim.lr_scheduler.CosineAnnealingLR):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

    return model, optimizer, lr_scheduler

def main():
    model, optimizer, criterion = initialize_model('resnet', num_classes=200, feature_extract = False)

    return model, optimizer, criterion


if __name__ == '__main__':
    main()