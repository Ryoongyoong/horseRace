'''-ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Modified by PARK, JunHo in April 10, 2022

[2] Writed by Jinsu, Kim in August 11, 2022

COPYRIGHT (c) 2022. CCNets. All Rights reserved.
'''
import os
import torch.utils.data

def _save_model(model_path, model_name, model, opt_model, scheduler_model):
    torch.save(model.state_dict(),os.path.join(model_path, model_name + '.pth'))
    torch.save(opt_model.state_dict(),os.path.join(model_path, 'opt_' + model_name + '.pth'))
    torch.save(scheduler_model.state_dict(),os.path.join(model_path, 'sch_' + model_name + '.pth'))

def _load_model(model_path, model_name, model, opt_model, scheduler_model):
    model.load_state_dict(torch.load(model_path + model_name + '.pth', map_location="cuda:0"))
    opt_model.load_state_dict(torch.load(model_path + 'opt_'+ model_name + '.pth', map_location="cuda:0"))
    opt_model.param_groups[0]['capturable'] = True
    scheduler_model.load_state_dict(torch.load(model_path + 'sch_'+ model_name + '.pth', map_location="cuda:0"))

def save_dataset(trainset, testset, path):
    if not os.path.isdir(path):
        os.makedirs(path)
    torch.save(trainset,path + "trainset.pt")
    torch.save(testset, path + "testset.pt")

def load_dataset(path):
    if not os.path.isdir(path):
        raise Exception(f"No such Path : {path}")
    trainset = torch.load(path + "trainset.pt")
    testset = torch.load(path + "testset.pt")
    return trainset, testset

def get_Dataloader(trainset, batch_size, shuffle = False, num_workers = 0, collate = None):
    return torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle = shuffle, num_workers = num_workers, collate_fn=collate)

def get_testloader(testset, batch_size, shuffle = False, num_workers = 0, collate = None):
    return torch.utils.data.DataLoader(testset, batch_size= batch_size, shuffle = shuffle, num_workers = num_workers, collate_fn=collate)

def get_evalloader(evalset, batch_size, num_workers = 0):
    return torch.utils.data.DataLoader(evalset, batch_size=batch_size, shuffle = False, num_workers = num_workers)

