'''
Written by SeuTao
'''
import os
import time
import numpy as np
import torch
from setting import parse_opts
from torch.utils.data import DataLoader
from datasets.TReNDs import TReNDsDataset
from model import generate_model
from tqdm import tqdm
#from apex import amp, optimizers

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


target_stats = {'age': [49.615210101624974, 13.539880871639827],
 'domain1_var1': [51.4744384335058, 10.188354350354668],
 'domain1_var2': [59.24640830885764, 11.38759506198513],
 'domain2_var1': [47.244965756472205, 11.124862686657542],
 'domain2_var2': [51.91631048336175, 11.839202896983359]}

def metric(y_true, y_pred):
    return np.mean(np.sum(np.abs(y_true - y_pred), axis=0) / np.sum(y_true, axis=0))

def weighted_nae(inp, targ):
    W = torch.FloatTensor([0.3, 0.175, 0.175, 0.175, 0.175])
    return torch.mean(torch.matmul(torch.abs(inp - targ), W.to(device) / torch.mean(targ, axis=0)))

def test(data_loader, model, sets, save_path):
    # settings
    print("validation")
    model.eval()

    y_pred = []
    ids_all = []
    with torch.no_grad():
        for batch_data in tqdm(data_loader):
                # getting data batch
                ids, volumes, feats, fncs, degs  = batch_data
                
                if not sets.no_cuda:
                    volumes = volumes.to(device)
                    feats = feats.to(device)
                    fncs = fncs.to(device)
                    degs = degs.to(device)
                    print(volumes.shape, feats.shape, fncs.shape, degs.shape)

                logits1, logits2 = model(volumes, feats, fncs, degs)
                
                features = ('age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2')
                for i in range(1):
                    logits1[:, i] *= target_stats[features[i]][1]
                    logits1[:, i] += target_stats[features[i]][0]
                
                y_pred.append(logits1.data.cpu().numpy())
                ids_all += ids

    y_pred = np.concatenate(y_pred, axis=0)
    np.savez_compressed(save_path,
                        y_pred = y_pred,
                        ids = ids_all)
    print(y_pred.shape)

import torch
import torch.nn as nn
import torch.nn.functional as F

class MishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.tanh(F.softplus(x))   # x * tanh(ln(1 + exp(x)))

    @staticmethod
    def backward(ctx, grad_output):
        x = ctx.saved_variables[0]
        sigmoid = torch.sigmoid(x)
        tanh_sp = torch.tanh(F.softplus(x)) 
        return grad_output * (tanh_sp + x * sigmoid * (1 - tanh_sp * tanh_sp))

class Mish(nn.Module):
    def forward(self, x):
        return MishFunction.apply(x)

def to_Mish(model):
    for child_name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, child_name, Mish())
        else:
            to_Mish(child)
    
    
if __name__ == '__main__':

    sets = parse_opts()
    sets.no_cuda = False
    #sets.resume_path = None
    sets.pretrain_path = None


    sets.model_name = r'prue_3dconv'
    sets.save_folder = r'./TReNDs/{}/' \
                       r'models_{}_{}_{}_fold_{}_feat_{}'.format(sets.model_name, 'resnet',sets.model_depth,
                                                                 sets.resnet_shortcut,sets.fold_index, sets.feat_index)

    if not os.path.exists(sets.save_folder):
        os.makedirs(sets.save_folder)

    # getting model
    torch.manual_seed(sets.manual_seed)
    model, parameters = generate_model(sets)
    model = model.to(device)
    to_Mish(model)
    print(model)
    print(device)

    # optimizer
    def get_optimizer(net):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=3e-4, betas=(0.9, 0.999), eps=1e-08)
        def ajust_lr(optimizer, epoch):
                if epoch < 24 :
                    lr = 3e-4
                elif epoch < 36:
                    lr = 1e-4
                else:
                    lr = 1e-5

                for p in optimizer.param_groups:
                    p['lr'] = lr
                return lr

        rate = ajust_lr(optimizer, 0)
        return  optimizer, ajust_lr

    optimizer, ajust_lr = get_optimizer(model)
    #model, optimizer = amp.initialize(model, optimizer,
    #                                  opt_level='O1',
    #                                  verbosity=0
    #                                  )
    model = torch.nn.DataParallel(model).to(device)
    print(sets.resume_path)
    # train from resume
    #i#f sets.resume_path:
        #if os.path.isfile(sets.resume_path):
    print("=> loading checkpoint '{}'".format(sets.resume_path))
    checkpoint = torch.load(f'{sets.save_folder}/{sets.resume_path}')
    model.load_state_dict(checkpoint['state_dict'])
        
    # getting data
    sets.phase = 'train'
    if sets.no_cuda:
        sets.pin_memory = False
    else:
        sets.pin_memory = True

    test_dataset = TReNDsDataset(mode='test', fold_index=sets.fold_index)
    test_loader  = DataLoader(test_dataset, batch_size=sets.batch_size,
                             shuffle=False, num_workers=sets.num_workers,
                              drop_last=False)
    test(test_loader, model, sets, sets.resume_path.replace('.pth.tar','.npz'))
