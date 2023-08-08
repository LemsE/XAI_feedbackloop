from torchvision import datasets, transforms
import torch
from typing import Tuple
from preprocess import mean, std, preprocess_input_function
from settings import  train_dir, test_dir, train_push_dir, \
                      test_batch_size, train_push_batch_size, train_batch_size,\
                      img_size


def create_imagefolders() -> datasets.ImageFolder:
    """
    Normalizes the data and creates ImageFolders to split the dataset in train, validation, and test

    Returns
    -----------
        image_datasets : ImageFolder
            ImageFolder of the train, push, and test set

    """ 
    normalize = transforms.Normalize(mean=mean,
                                    std=std)
    image_datasets = {
    'train': datasets.ImageFolder(
    train_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])),
    'push':datasets.ImageFolder(
    train_push_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
    ])),
    'test':datasets.ImageFolder(
    test_dir,
    transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ]))
    }

    return image_datasets

def create_dataloaders() -> torch.utils.data.DataLoader:
    """
    Creates DataLoaders from the ImageFolders into a train, validation, and test dataloader

    Returns
    -----------
        dataloaders : Dataloader
            Dataloader used for loading the data onto the ResNet50 model

    """ 
    image_datasets = create_imagefolders()

    data_loaders = {
    'train': torch.utils.data.DataLoader(
    image_datasets['train'], batch_size=train_batch_size, shuffle=True,
    num_workers=4, pin_memory=False),
    'push': torch.utils.data.DataLoader(
    image_datasets['push'], batch_size=train_push_batch_size, shuffle=False,
    num_workers=4, pin_memory=False),
    'test': torch.utils.data.DataLoader(
    image_datasets['test'], batch_size=test_batch_size, shuffle=False,
    num_workers=4, pin_memory=False)
    }

    print('Dataloaders created')

    return data_loaders, image_datasets


def main():
    # Just for testing
    dataloaders = create_dataloaders()
    return dataloaders

if __name__ == '__main__':
    main()