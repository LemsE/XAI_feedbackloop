import preprocessing_resnet
import train_and_test_resnet as tnt
import model_resnet
import torch
import yaml




def main():
    # Will be cleaned up
    model_name = 'resnet34'
    num_classes = 200
    save_path = './Resnet50/trained_models'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ckp_n =input('Which checkpoint do you want to use? (int): ')
    checkpoint_path='/home/azureuser/cloudfiles/code/Users/elise.lems/Explaining_Prototypes_XAI_loop/Resnet50/trained_models/checkpoint_{0}.pt'.format(ckp_n)

    with open('/home/azureuser/cloudfiles/code/Users/elise.lems/Explaining_Prototypes_XAI_loop/Resnet50/conf/config.yaml') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # Create datatloaders
    dataloaders, train_set, valid_set, test_set = preprocessing_resnet.main()
    # Initialize the model 
    ResNet, optimizer, criterion, scheduler = model_resnet.initialize_model(lr=1e-3,model_name=model_name, num_classes=num_classes, feature_extract = False)
    

    loaded_model, optimizer, scheduler = model_resnet.load_ckp(checkpoint_path=checkpoint_path, model=ResNet, optimizer=optimizer, lr_scheduler=scheduler)

    # # Test a trained model
    tnt.test(loaded_model=loaded_model, device=device, dataloaders=dataloaders, test_set=test_set)

    




if __name__ == '__main__':
    main()
    