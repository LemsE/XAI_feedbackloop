import os
import torch
import shutil

def save_model_w_condition(model, model_dir, model_name, accu, target_accu, log=print):
    '''
    model: this is not the multigpu model
    '''
    if accu > target_accu:
        log('\tabove {0:.2f}%'.format(target_accu * 100))
        # torch.save(obj=model.state_dict(), f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))
        torch.save(obj=model, f=os.path.join(model_dir, (model_name + '{0:.4f}.pth').format(accu)))

def save_ckp(state,  checkpoint_dir):
    makedir(checkpoint_dir)
    epoch_n = state['epoch']
    f_path = checkpoint_dir + 'checkpoint_10c{0}.pt'.format(epoch_n)
    torch.save(state, f_path)
    


def load_ckp(checkpoint_path, model, joint_optimizer, warm_optimizer, last_layer_optimizer, joint_lr_scheduler):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    joint_optimizer.load_state_dict(checkpoint['joint_optimizer'])
    warm_optimizer.load_state_dict(checkpoint['warm_optimizer'])
    last_layer_optimizer.load_state_dict(checkpoint['last_layer_optimizer'])
    joint_lr_scheduler.load_state_dict(checkpoint['joint_lr_scheduler'])


    return model, joint_optimizer,warm_optimizer ,last_layer_optimizer,joint_lr_scheduler,checkpoint['epoch']

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)   