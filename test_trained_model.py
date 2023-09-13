from helpers import list_of_distances, make_one_hot
from tqdm import tqdm
from load_data_ppnet import create_dataloaders
from settings import base_architecture, img_size, prototype_shape, num_classes, \
                     prototype_activation_function, add_on_layers_type, experiment_run, \
                     train_batch_size,  \
                     joint_optimizer_lrs, joint_lr_step_size, warm_optimizer_lrs, last_layer_optimizer_lr, \
                     coefs, num_train_epochs, num_warm_epochs, push_start, push_epochs, test_start, test_epochs, labels_plot

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import train_and_test as tnt
torch.manual_seed(42)
from sklearn.metrics import top_k_accuracy_score, precision_score, recall_score, f1_score

import random
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)
"""
Code for testing intermediate checkpoints

"""

def test(
        loaded_model, device, 
        dataloaders, img_sets
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
    start = timeit.default_timer()
    loaded_model.eval()
    with torch.no_grad():
        n_correct = 0
        for inputs,labels in tqdm(dataloaders['test']):
            inputs = inputs.to(device=device)
            labels = labels.to(device=device)

            output, min_dist = loaded_model(inputs)
            __, predicted = torch.max(output.data, 1)
            
            n_correct += torch.sum(predicted == labels.data)
        accuracy = n_correct/len(dataloaders['test'].dataset)
        print('Test accuracy: {:.4f} '.format(accuracy))
    stop = timeit.default_timer()
    time = stop - start
    print('Test time in s: {} '.format(time) )

def get_misclassified(loaded_model, device, 
        dataloaders):
    loaded_model.eval()
    with torch.no_grad():
        n_correct = 0
        mis_samples = []
        for inputs,labels in tqdm(dataloaders['test']):
            inputs = inputs.to(device=device)
            labels = labels.to(device=device)

            output, min_dist = loaded_model(inputs)
            __, predicted = torch.max(output.data, 1)

            mask = (predicted!=labels)
            missclassified_img = inputs[mask]
            missclassified_lab = labels[mask]
            mis_samples.append(missclassified_img)

            n_correct += torch.sum(predicted == labels.data)

        accuracy = n_correct/len(dataloaders['test'].dataset)
        print('Test accuracy: {:.4f} '.format(accuracy))
        print(mis_samples)

def conf_matrix(loaded_model, device, 
        dataloaders):
        loaded_model.eval()
        with torch.no_grad():
            n_correct = 0
            # mis_samples = []
            true_labels = []
            pred_labels = []
            logits = []
            # save_labels = None
            for inputs,labels in tqdm(dataloaders['test']):
                # save_labels = labels.
                inputs = inputs.to(device=device)
                labels = labels.to(device=device)

                output, min_dist = loaded_model(inputs)
                __, predicted = torch.max(output.data, 1)
                
                true_labels.extend(labels.tolist())
                pred_labels.extend(predicted.tolist())
                logits.extend(output.tolist())

                n_correct += torch.sum(predicted == labels.data)

            accuracy = n_correct/len(dataloaders['test'].dataset)
            print('Test accuracy: {:.4f} '.format(accuracy))
            true_labels = np.array(true_labels)
            np.savetxt("true labels.csv", true_labels, delimiter=",", fmt ='% s')
            predicted_labels = np.array(pred_labels)
            np.savetxt("predicted_labels.csv", predicted_labels, delimiter=",", fmt ='% s')
            logits = np.array(logits)
            print("Top 5 accuracy: ", top_k_accuracy_score(true_labels, logits, k=5,labels=np.arange(0, 20, 1)))

            print('precision score: ', precision_score(true_labels, predicted_labels, average='weighted'))
            print('recall score: ', recall_score(true_labels, predicted_labels, average='weighted'))
            print('f1 score: ', f1_score(true_labels, predicted_labels, average='weighted'))

            confusion_mat = confusion_matrix(true_labels, predicted_labels)
            cm_display = metrics.ConfusionMatrixDisplay(confusion_mat, display_labels=labels_plot)
            cm_display.plot()
            plt.show()
            
            df = pd.DataFrame(confusion_matrix, index=true_labels, columns=true_labels)


def test_conv_features(loaded_model, device, dataloaders):
    start = timeit.default_timer()
    loaded_model.eval()
    with torch.no_grad():
        n_correct = 0
        for inputs,labels in tqdm(dataloaders['test']):
            inputs = inputs.to(device=device)
            labels = labels.to(device=device)

            x = loaded_model.module.conv_features(inputs)
            print(x.size())

    stop = timeit.default_timer()
    time = stop - start
    print('Test time in s: {} '.format(time) )





        
def main():

    dataloaders, img_sets = create_dataloaders()

    load_model_path = './models/beta09/seeds/24_9end0.8272.pth' 
    ppnet = model.construct_PPNet(base_architecture=base_architecture,
                                pretrained=True, img_size=img_size,
                                prototype_shape=prototype_shape,
                                num_classes=num_classes,
                                prototype_activation_function=prototype_activation_function,
                                add_on_layers_type=add_on_layers_type)
    

    
    ppnet = torch.load(load_model_path)
    ppnet = ppnet.cuda()
    ppnet_multi = torch.nn.DataParallel(ppnet)

    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf_matrix(ppnet_multi, device, dataloaders)
    #test(loaded_model=ppnet_multi,device=device,dataloaders=dataloaders, img_sets=img_sets)



if __name__ == '__main__':
    main()

