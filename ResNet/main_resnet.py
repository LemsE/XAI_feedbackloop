import preprocessing_resnet
import train_and_test_resnet as tnt
import model_resnet
import torch
import numpy as np
import random

def main():
    seed = 5512 

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    model_name = 'resnet34'
    num_classes = 20
    save_path = '/home/azureuser/cloudfiles/code/Users/elise.lems/Explaining_Prototypes_XAI_loop/Resnet50/trained_models/'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    

    # Create datatloaders
    dataloaders, train_set, valid_set, test_set = preprocessing_resnet.main()

    ResNet, optimizer, criterion, scheduler = model_resnet.initialize_model(lr=1e-3,model_name=model_name, num_classes=num_classes, feature_extract = True)
    trained_model = tnt.train_model(model=ResNet, criterion=criterion, optimizer=optimizer, 
                                            lr_scheduler=scheduler, device=device, dataloaders=dataloaders, train_set=train_set, valid_set=valid_set, num_epochs = 12)

    tnt.test(loaded_model=trained_model, device=device, dataloaders=dataloaders, test_set=test_set)
    


if __name__ == '__main__':
    main()
    