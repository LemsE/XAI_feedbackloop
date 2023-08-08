import os
import shutil

import torch
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np

import argparse
import re

from helpers import makedir
import model
import push

import train_and_test as tnt
import save
from log import create_logger
from preprocess import mean, std, preprocess_input_function

# book keeping namings and code
from settings import base_architecture, img_size, prototype_shape, num_classes, \
                     prototype_activation_function, add_on_layers_type, experiment_run

# load the data
from settings import train_dir, test_dir, train_push_dir, \
                     train_batch_size, test_batch_size, train_push_batch_size
# define optimizer
from settings import joint_optimizer_lrs, joint_lr_step_size

# weighting of different training losses
from settings import coefs

# number of training epochs, number of warm epochs, push start epoch, push epochs
from settings import num_train_epochs, num_warm_epochs, push_start, push_epochs, beta
from settings import last_layer_optimizer_lr
from load_data_ppnet import create_dataloaders

# reward model
import sys

sys.path.append('./Reward_model')
from reward_model import RewardModel

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0') # python3 main.py -gpuid=0,1,2,3
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpuid[0]
print(os.environ['CUDA_VISIBLE_DEVICES'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Get the base architecture
base_architecture_type = re.match('^[a-z]*', base_architecture).group(0)

# Prepare saving directory
model_dir = './saved_models/' + base_architecture + '/' + experiment_run + '/'
makedir(model_dir)
shutil.copy(src=os.path.join(os.getcwd(), __file__), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'settings.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), base_architecture_type + '_features.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'model.py'), dst=model_dir)
shutil.copy(src=os.path.join(os.getcwd(), 'train_and_test.py'), dst=model_dir)

log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
img_dir = os.path.join(model_dir, 'img')
makedir(img_dir)
weight_matrix_filename = 'outputL_weights'
prototype_img_filename_prefix = 'prototype-img'
prototype_self_act_filename_prefix = 'prototype-self-act'
proto_bound_boxes_filename_prefix = 'bb'


# Prepare and load data
normalize = transforms.Normalize(mean=mean,
                                 std=std)

data_loaders, image_datasets = create_dataloaders()
train_loader = data_loaders['train']
test_loader = data_loaders['test']
train_push_loader = data_loaders['push']

log('training set size: {}'.format(len(train_loader.dataset)))
log('test set size: {}'.format(len(test_loader.dataset)))
log('push set size: {}'.format(len(train_push_loader.dataset)))
log('batch size: {0}'.format(train_batch_size))





# Reward model
input_size_1 = (1,7,7)
input_size_2 = (3,224,224)


load_model_path_rm = './Reward_model/trained_models/weights_reward_01.h5' 
reward_model = RewardModel(input_size_1,input_size_2)
reward_model.load_state_dict(torch.load(load_model_path_rm))
reward_model.to(device)
# Freeze reward model
for param in reward_model.parameters():
    param.requires_grad = False


# construct the model
ppnet = model.construct_PPNet(base_architecture=base_architecture,
                              pretrained=True, img_size=img_size,
                              prototype_shape=prototype_shape,
                              num_classes=num_classes,
                              prototype_activation_function=prototype_activation_function,
                              add_on_layers_type=add_on_layers_type)

ppnet = ppnet.cuda()
ppnet_multi = torch.nn.DataParallel(ppnet)
class_specific = True


optimizer_specs = \
[{'params': ppnet.features.parameters(), 'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
 {'params': ppnet.add_on_layers.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
 {'params': ppnet.prototype_vectors, 'lr': joint_optimizer_lrs['prototype_vectors']},
]
optimizer = torch.optim.Adam(optimizer_specs)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=joint_lr_step_size, gamma=0.1)


last_layer_optimizer_specs = [{'params': ppnet.last_layer.parameters(), 'lr': last_layer_optimizer_lr}]
last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)


# train the model
log('start training')
# for plotting
train_accuracy_values = []
train_loss_values = []
test_accuracy_values = []

for epoch in range(num_train_epochs):
    log('epoch: \t{0}'.format(epoch))

    

    tnt.feedback_loop(model=ppnet_multi, log=log)

    accu_train, total_loss_train = tnt.train_feedbackloop(model=ppnet_multi, reward_model=reward_model,dataloader=train_loader, optimizer=optimizer,
                    class_specific=class_specific, coefs=coefs, log=log, beta = beta)
    lr_scheduler.step()
    train_accuracy_values.append(accu_train)
    train_loss_values.append(total_loss_train)

    accu_test,_,_ = tnt.test(model=ppnet_multi, dataloader=test_loader,
                    class_specific=class_specific, log=log)
    test_accuracy_values.append(accu_test)
    save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'nopush', accu=accu_test,
                                target_accu=0.80, log=log)

    if epoch >= push_start and epoch in push_epochs:
        push.push_prototypes(
            train_push_loader, # pytorch dataloader (must be unnormalized in [0,1])
            prototype_network_parallel=ppnet_multi, # pytorch network with prototype_vectors
            class_specific=class_specific,
            preprocess_input_function=preprocess_input_function, # normalize if needed
            prototype_layer_stride=1,
            root_dir_for_saving_prototypes=img_dir, # if not None, prototypes will be saved here
            epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
            save_prototype_class_identity=True,
            log=log)
        accu, _, _ = tnt.test(model=ppnet_multi, dataloader=test_loader,
                        class_specific=class_specific, log=log)
        save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=accu,
                                    target_accu=0.80, log=log)
        
        if prototype_activation_function != 'linear':
            tnt.last_only(model=ppnet_multi, log=log)
            for i in range(20):
                log('iteration: \t{0}'.format(i))
                _,_,_ = tnt.train(model=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer,
                                class_specific=class_specific, coefs=coefs, log=log)
                accu,_,_ = tnt.test(model=ppnet_multi, dataloader=test_loader,
                                class_specific=class_specific, log=log)
                save.save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + '_' + str(i) + 'push', accu=accu,
                                            target_accu=0.80, log=log)

   

log('Saving accuracies and losses...')
train_accuracy_values_arr = np.array(train_accuracy_values)
train_loss_values_arr = np.array(train_loss_values)
test_accuracy_values_arr = np.array(test_accuracy_values)


np.savetxt('train_acc.csv', train_accuracy_values_arr, delimiter=',')
np.savetxt('train_loss.csv', train_loss_values_arr, delimiter=',')
np.savetxt('test_acc.csv', test_accuracy_values_arr, delimiter=',')

log('Saved!')

logclose()