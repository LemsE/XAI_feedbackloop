import model_resnet
import torch
import preprocessing_resnet
import os
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torchvision import  models, datasets
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
import torchvision.models
import timeit
from early_stopping_resnet import EarlyStopping_resnet
from tqdm import tqdm
import mlflow
from sklearn.metrics import top_k_accuracy_score, precision_score, recall_score, f1_score
import numpy as np
"""
Todo:
- Add logger
- Add early stopping
- Add fine tuning
"""

def train_model(
        model : torchvision.models.ResNet, criterion : torch.nn.modules.loss.CrossEntropyLoss, optimizer : torch.optim.AdamW, lr_scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
        device : torch.device, dataloaders : torch.utils.data.DataLoader, 
        train_set : torch.utils.data.dataset.Subset, valid_set : torch.utils.data.dataset.Subset, num_epochs : int = 5
            ) -> Tuple[torchvision.models.ResNet, pd.DataFrame]:
    """
    Train a PyTorch model for a specified num_epochs. Returns a trained model and the history of the model.
    
    Parameters
    -----------
        model : torchvision.models.ResNet
            Pretrained ResNet model 
        criterion : torch.nn.modules.loss.CrossEntropyLoss
            Loss function for learning criterion: cross entropy loss
        optimizer : torch.optim.Adam
            Adam optimizer
        device : torch.device
            Device to which the model is send
        dataloaders : Dataloader
            Dataloader used for loading the data onto the ResNet50 model
        train_set : Subset
            Training dataset 
        valid_set : Subset
            Validation dataset
        num_epochs : int
            Number of epochs used for training
    Returns
    -----------
        model : torchvision.models.ResNet
            Trained ResNet model
        history : pd.DataFrame
            Dataframe containing the train and validation losses of the model

    """
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    train_accuracy_lst = []
    train_loss_lst = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        dataset_size = 0
        for phase in ['train']:
            start = timeit.default_timer()

            dataset_size = len(train_set)
            model.train()

            running_loss = 0.0
            running_corrects = 0

            for i, (inputs, labels) in enumerate(tqdm(dataloaders[phase])):

                inputs = inputs.to(device=device)
                labels = labels.to(device=device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            lr_scheduler.step()

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            acc = epoch_acc.cpu()
            acc = np.array(acc)
            # history.loc[epoch, 'accu'] = float(acc)
            train_accuracy_lst.append(acc)
            train_loss_lst.append(epoch_loss)

            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                        epoch_loss,
                                                        epoch_acc))
            
        
            stop = timeit.default_timer()
            time = stop - start
            print('{} time in s: {} '.format(phase, time) )
    # save model
    # checkpoint ={
    #                 'epoch': epoch+1,
    #                 'state_dict': model.state_dict(),
    #                 'optimizer': optimizer.state_dict(),
    #                 'lr_scheduler': lr_scheduler.state_dict()
                # }
    save_model(model, save_path='./trained_models/')
    print('Model saved')
    # history['accu'] = history['accu'].astype(float)
    # plot_losses(history)

    # np.savetxt("accuracy_test1.csv",
    #     train_accuracy_lst,
    #     delimiter =", ",
    #     fmt ='% s')
    # np.savetxt("loss_test1.csv",
    #     train_loss_lst,
    #     delimiter =", ",
    #     fmt ='% s')

            
    return model


def makedir(path) -> None:
    """
    Creates a new folder in the specified path if the folder with name folder_name does not exist yet.

    Parameters
    -----------
        path : str
            Destination path in which the folder will be made
    """
    try: 
        # mode of the folder (default)
        mode = 0o777
        # Create the new folder
        os.mkdir(path, mode)
        print("Directory '% s' is built!" % path)  
    except OSError as error: 
        print(error)


def save_model(trained_model : torchvision.models.ResNet, save_path : str) -> None:
    """
    Makes a directory in the specified save_path and saves the weights of a model.

    Parameters
    -----------
        trained_model : torchvision.models.ResNet
            Trained ResNet model 
        save_path : str
            Destination path in which the folder will be made

    """
    makedir(save_path)

    torch.save(trained_model.state_dict(), save_path + 'weights_resnet34_seed1.h5')



def test(
        loaded_model : torchvision.models.ResNet, device : torch.device,
        dataloaders : torch.utils.data.DataLoader, test_set : datasets.ImageFolder
        ) -> None:
    """
    Tests a PyTorch model. 

    Parameters
    -----------
        loaded_model : torchvision.models.ResNet
            Trained and loaded ResNet model
        device : torch.device
            Device to be utilized
        dataloaders : Dataloader
            Dataloader used for loading the data onto the model
        test_set : datasets.ImageFolder
            Test set of the data
    
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    start = timeit.default_timer()
    loaded_model.eval()
    with torch.no_grad():
        true_labels = []
        logits = []
        pred_labels = []
        n_correct = 0
        for (inputs,labels) in tqdm(dataloaders['test']):
            inputs = inputs.to(device=device)
            labels = labels.to(device=device)

            outputs = loaded_model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            n_correct += torch.sum(predicted == labels.data)

            true_labels.extend(labels.tolist())
            pred_labels.extend(predicted.tolist())
            logits.extend(outputs.tolist())
        accuracy = n_correct/len(test_set)

        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels)
        print('Test accuracy: {:.4f} '.format(accuracy))
        print("Top 5 accuracy: ", top_k_accuracy_score(true_labels, logits, k=5,labels=np.arange(0, 20, 1)))
        print('precision score: ', precision_score(true_labels, pred_labels, average='weighted'))
        print('recall score: ', recall_score(true_labels, pred_labels, average='weighted'))
        print('f1 score: ', f1_score(true_labels, pred_labels, average='weighted'))

    stop = timeit.default_timer()
    time = stop - start
    print('Test time in s: {} '.format(time) )


def plot_losses(history : pd.DataFrame) -> None:
    """
    Plots the losses from history

    Parameters
    -----------
        history : pd.DataFrame
            Dataframe containing the train and validation losses of the model
    
    """
    history.plot( y= ['accu'], kind='line', figsize=(7,4))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Train accuracy')
    plt.legend()
    plt.show()
    plt.savefig('history_resnet.png')



def main():
    ResNet, optimizer, criterion, _= model_resnet.initialize_model(lr=1e-4, model_name='resnet34', num_classes=20, feature_extract = False)
    dataloaders, train_set, valid_set, test_set = preprocessing_resnet.main()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    trained_model, history = train_model(ResNet, criterion, optimizer, device, dataloaders, train_set, valid_set, num_epochs = 5)

    save_path = './trained_models/'

    # save_model(trained_model=trained_model, save_path=save_path)

    # loaded_model = model_resnet.load_model(model ='resnet34',device=device, path=save_path + 'weights_resnet34_test_plus5.h5', num_classes=20)
    # loaded_model.to(device=device)
    test(loaded_model=trained_model, device=device, dataloaders=dataloaders, test_set=test_set)

    #plot_losses(history=history)



if __name__ == '__main__':
    main()