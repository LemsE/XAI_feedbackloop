base_architecture = 'resnet34'
img_size = 224
prototype_shape = (200, 128, 1, 1)
num_classes = 20
prototype_activation_function = 'log'
add_on_layers_type = 'regular'

experiment_run = 'beta1_p200'

data_path = './data/'
train_push_dir = data_path + 'push_first20/'
train_dir = data_path + 'train_first20/'
test_dir = data_path + 'test_first20/'
# train_dir = data_path + 'train_sub10/'
# test_dir = data_path + 'test_sub10/'
# train_push_dir = data_path + 'train_push10/'

# train_push_dir = data_path + 'train_cropped/'
train_batch_size = 128
test_batch_size = 100
train_push_batch_size = 75

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}
joint_lr_step_size = 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    'clst': 0.8,
    'sep': -0.08,
    'l1': 1e-4,
}

num_train_epochs = 25
num_warm_epochs = 5

push_start = 5
push_epochs = [i for i in range(num_train_epochs) if i % 5 == 0]

test_start = 2
test_epochs = [i for i in range(num_train_epochs) if i % 2 == 0]

import numpy as np
labels_plot = np.arange(0, 20, 1)

beta = 1