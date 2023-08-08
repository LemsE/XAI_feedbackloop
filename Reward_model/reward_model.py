import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import torch.nn.functional as F

class RewardModel(nn.Module):
    def __init__(self, input1, input2):
        super().__init__()
         # Feature layers for input 1 (prototype)
        self.features1 = nn.Sequential(
            nn.Conv2d(input1[0], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Feature layers for input 2 (input image x)
        self.features2 = nn.Sequential(
            nn.Conv2d(input2[0], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Regression layers
        self.regression = nn.Sequential(
            nn.Linear((64*3*3)+(128*56*56),128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
    def forward(self, x1, x2):
        x1 = self.features1(x1)

        x2 = self.features2(x2)
        # print("X1 before", x1.size())
        # print("X2 before", x2.size())
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        # print("X1", x1.size())
        # print("X2", x2.size())
   
        x = torch.cat((x1, x2), dim=1)

        x = self.regression(x)

        return x

    # def __init__(self, num_classes):
    #     super().__init__()
    #     self.features = nn.Sequential(
    #         nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
    #         nn.ReLU(inplace=True),
    #         nn.MaxPool2d(kernel_size=2, stride=2),
    #         nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
    #         nn.ReLU(inplace=True),
    #         nn.MaxPool2d(kernel_size=2, stride=2)
    #     )

    #     self.classifier = nn.Sequential(
    #         nn.Linear(32 * 56 * 56, 512),
    #         nn.ReLU(inplace=True),
    #         nn.Linear(512, num_classes)
    #     )

    # def forward(self, x):
    #     x = self.features(x)
    #     x = torch.flatten(x, 1)
    #     x = self.classifier(x)
    #     return x
    
        # self.features = nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )

        # self.classifier = nn.Sequential(
        #     nn.Linear(128 * 14 * 14, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, num_classes)
        # )

