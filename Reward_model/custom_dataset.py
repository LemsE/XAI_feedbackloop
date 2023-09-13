import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data_file, image_folder_prototype, image_folder_inputx,transform=None):
        self.data = pd.read_csv(data_file, sep=',', header=None)
        self.image_folder_p = image_folder_prototype
        self.image_folder_inputx = image_folder_inputx
        self.transform = transform
        

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        image_filename, rank = self.data.iloc[idx]
        rank = torch.tensor(int(rank))
        image_path_p = os.path.join(self.image_folder_p, image_filename+ '.npy')
        image_path_inputx = os.path.join(self.image_folder_inputx, image_filename + '.jpg')

        prototype = np.load(image_path_p)
        prototype = torch.from_numpy(prototype)
        inputx = Image.open(image_path_inputx)

        if self.transform is not None:
            inputx = self.transform(inputx)
            prototype = torch.unsqueeze(prototype, dim=0)

        return prototype, rank, inputx # prototype, rank, input image