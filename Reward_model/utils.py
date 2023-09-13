
import torch
import os
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torchvision import  models, datasets
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
import torchvision.models
import numpy as np



def save_model(trained_model, save_path) -> None:
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

    torch.save(trained_model.state_dict(), save_path + 'weights_reward_01.h5')



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