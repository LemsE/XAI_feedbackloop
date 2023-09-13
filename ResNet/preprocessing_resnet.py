from torchvision import datasets, models, transforms
import torch
import torchvision
from typing import Tuple


def create_imagefolders(
        train_dir : str, test_dir : str, 
        mean : float, std : float, img_size : int
        ) -> Tuple[datasets.ImageFolder, torch.utils.data.dataset.Subset, torch.utils.data.dataset.Subset]:
    """
    Normalizes the data and creates ImageFolders to split the dataset in train, validation, and test

    Parameters
    -----------
        train_dir : str
            Directory where all train images are stored
        test_dir : str
            Directory where all test images are stored
        mean : float
            Mean value of the images (taken from settings.py file)
        std : float
            Standard deviation value of the images (taken from settings.py file)
        img_size : int
            Image size (taken from settings.py file)

    Returns
    -----------
        image_datasets : ImageFolder
            ImageFolder of the train and test set
        train_set : Subset
            Training dataset 
        valid_set
            Validation dataset

    """ 
    normalize = transforms.Normalize(mean=mean,
                                    std=std)
    image_datasets = {
        'train':datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ])),
        'test':
        datasets.ImageFolder(
            test_dir,
            transforms.Compose([
                transforms.Resize(size=(img_size, img_size)),
                transforms.ToTensor(),
                normalize,
            ]))
    }

    train_set_size = int(len(image_datasets['train']) * 0.95)
    #vali_2 = train_set_size *0.8
    valid_set_size = len(image_datasets['train']) - train_set_size
    train_set, valid_set = torch.utils.data.random_split(image_datasets['train'],[train_set_size,valid_set_size])
    # train_set_size_1 = int(len(train_set_0) * 0.5)
    # valid_set_size_1 = len(train_set_0) - train_set_size_1 
    # train_set , valid_set = torch.utils.data.random_split(image_datasets['train'],[train_set_size_1,valid_set_size_1])

    print('Train set size: ',(len(image_datasets['train'])))
    #print('Validation set size: ',(len(valid_set)))
    print('Test set size: ',(len(image_datasets['test'])))


    return image_datasets, train_set, valid_set

def create_dataloaders(
        train_set : torch.utils.data.dataset.Subset, valid_set : torch.utils.data.dataset.Subset, 
        image_datasets : datasets.ImageFolder, 
        train_batch_size : int, test_batch_size : int
        ) -> torch.utils.data.DataLoader:
    """
    Creates DataLoaders from the ImageFolders into a train, validation, and test dataloader

    Parameters
    -----------
        image_datasets : ImageFolder
            ImageFolder of the train and test set
        train_set : Subset
            Training dataset 
        valid_set : Subset
            Validation dataset
        train_batch_size : int
            Batch size for train data
        test_batch_size : int
            Batch size for test data

    Returns
    -----------
        dataloaders : Dataloader
            Dataloader used for loading the data onto the ResNet50 model

    """ 
    g = torch.Generator()
    g.manual_seed(42)

    dataloaders = {
    'train':
    torch.utils.data.DataLoader(
    image_datasets['train'], batch_size=train_batch_size, shuffle=True,
    num_workers=4, pin_memory=False, generator=g),
    # 'validation':
    # torch.utils.data.DataLoader(
    # valid_set, batch_size=train_batch_size, shuffle=True,
    # num_workers=4, pin_memory=False),
    'test':
    torch.utils.data.DataLoader(
    image_datasets['test'], batch_size=test_batch_size, shuffle=False,
    num_workers=4, pin_memory=False, generator=g)
    }

    print('Dataloaders created')

    return dataloaders


def main():
    # This will be cleaned up
    train_dir = './data/CUB_200_2011/datasets/cub200_cropped/train_sub10'
    test_dir = './data/CUB_200_2011/datasets/cub200_cropped/test_sub10'

    img_size = 224
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    train_batch_size = 256
    test_batch_size = 128

    image_datasets, train_set, valid_set = create_imagefolders(train_dir=train_dir, test_dir=test_dir, mean=mean, std=std, img_size=img_size)
    dataloaders = create_dataloaders(train_set=train_set, valid_set=valid_set, image_datasets=image_datasets, train_batch_size=train_batch_size, test_batch_size=test_batch_size)
    # for i, (_,_) in enumerate(dataloaders['train']):
    #     print(i)
    return dataloaders, train_set, valid_set, image_datasets['test']

if __name__ == '__main__':
    main()
    

