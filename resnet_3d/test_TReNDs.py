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

def metric(y_true, y_pred):
    return np.mean(np.sum(np.abs(y_true - y_pred), axis=0) / np.sum(y_true, axis=0))

def weighted_nae(inp, targ):
    W = torch.FloatTensor([0.3, 0.175, 0.175, 0.175, 0.175])
    return torch.mean(torch.matmul(torch.abs(inp - targ), W.to(device) / torch.mean(targ, axis=0)))

def valid(data_loader, model, sets):
    # settings
    print("validation")
    model.eval()

    y_pred = []
    y_true = []
    loss_ave = []

    with torch.no_grad():
        for batch_data in tqdm(data_loader):
            # getting data batch
            volumes, feats, fncs, degs, label = batch_data
            if not sets.no_cuda:
                volumes = volumes.to(device)
                feats = feats.to(device)
                fncs = fncs.to(device)
                degs = degs.to(device)
                label = label.to(device)

                logits = model(volumes, feats, fncs, degs)

                # calculating loss
                loss_value = weighted_nae(logits, label)
                y_pred.append(logits.data.cpu().numpy())
                y_true.append(label.data.cpu().numpy())
                loss_ave.append(loss_value.data.cpu().numpy())

    print('valid loss', np.mean(loss_ave))
    y_pred = np.concatenate(y_pred,axis=0)
    y_true = np.concatenate(y_true,axis=0)

    domain = ['age', 'domain1_var1', 'domain1_var2', 'domain2_var1', 'domain2_var2']
    w = [0.3, 0.175, 0.175, 0.175, 0.175]

    m_all = 0
    for i in range(5):
        m = metric(y_true[:,i], y_pred[:,i])
        print(domain[i],'metric:', m)
        m_all += m*w[i]

    print('all_metric:', m_all)
    model.train()
    return np.mean(loss_ave)

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

                logits = model(volumes, feats, fncs, degs)
                
                y_pred.append(logits.data.cpu().numpy())
                ids_all += ids

    y_pred = np.concatenate(y_pred, axis=0)
    np.savez_compressed(save_path,
                        y_pred = y_pred,
                        ids = ids_all)
    print(y_pred.shape)

def train(train_loader,valid_loader, model, optimizer, total_epochs, save_interval, save_folder, sets):
    f = open(os.path.join(save_folder,'log.txt'),'w')

    # settings
    batches_per_epoch = len(train_loader)
    print("Current setting is:")
    print(sets)
    print("\n\n")

    model.train()
    train_time_sp = time.time()

    valid_loss = 99999
    min_loss = 99999

    for epoch in range(total_epochs):
        rate = adjust_learning_rate(optimizer, epoch)

        # Training
        # log.info('lr = {}'.format(scheduler.get_lr()))
        tk0 = tqdm(train_loader, total=int(len(train_loader)))
        for batch_id, batch_data in enumerate(tk0):
            # getting data batch
            batch_id_sp = epoch * batches_per_epoch
            volumes, feats, fncs, degs, label = batch_data

            if not sets.no_cuda: 
                volumes = volumes.to(device)
                feats = feats.to(device)
                fncs = fncs.to(device)
                degs = degs.to(device)
                label = label.to(device)

            optimizer.zero_grad()
            logits = model(volumes, feats, fncs, degs)

            # calculating loss
            loss = weighted_nae(logits, label)
            #with amp.scale_loss(loss, optimizer) as scaled_loss:
            #    scaled_loss.backward()
            loss.backward()                
            optimizer.step()

            avg_batch_time = (time.time() - train_time_sp) / (1 + batch_id_sp)

            log_ = '{} Batch: {}-{} ({}), ' \
                   'lr = {:.5f}, ' \
                   'train loss = {:.3f}, ' \
                   'valid loss = {:.3f}, ' \
                   'avg_batch_time = {:.3f} '.format(sets.model_name, epoch, batch_id, batch_id_sp, rate, loss.item(), valid_loss, avg_batch_time)

            #print(log_)
            f.write(log_ + '\n')
            f.flush()

       # valid
        valid_loss = valid(valid_loader,model,sets)

        if valid_loss < min_loss:
            min_loss = valid_loss
            model_save_path = '{}/epoch_{}_batch_{}_loss_{}.pth.tar'.format(save_folder, epoch, batch_id, valid_loss)

            model_save_dir = os.path.dirname(model_save_path)
            if not os.path.exists(model_save_dir):
                os.makedirs(model_save_dir)

            log_ = 'Save checkpoints: epoch = {}, batch_id = {}'.format(epoch, batch_id)
            print(log_)
            f.write(log_ + '\n')

            torch.save({'epoch': epoch,
                                'batch_id': batch_id,
                                'state_dict': model.state_dict(),
                                'optimizer': optimizer.state_dict()},
                                model_save_path)

    print('Finished training')
    f.close()

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
                       r'models_{}_{}_{}_fold_{}'.format(sets.model_name, 'resnet',sets.model_depth,sets.resnet_shortcut,sets.fold_index)

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

#     train_dataset = TReNDsDataset(mode='train', fold_index=sets.fold_index)
#     train_loader = DataLoader(train_dataset, batch_size=sets.batch_size,
#                              shuffle=True, num_workers=sets.num_workers,drop_last=True)

#     valid_dataset = TReNDsDataset(mode='valid', fold_index=sets.fold_index)
#     valid_loader = DataLoader(valid_dataset, batch_size=sets.batch_size,
#                              shuffle=False, num_workers=sets.num_workers, drop_last=False)

#     # # training
#     train(train_loader, valid_loader,model, optimizer,ajust_lr,
#           total_epochs=sets.n_epochs,
#           save_interval=sets.save_intervals,
#           save_folder=sets.save_folder, sets=sets)

    # # validate
    #valid(valid_loader, model, sets)
    test_dataset = TReNDsDataset(mode='test', fold_index=sets.fold_index)
    test_loader  = DataLoader(test_dataset, batch_size=sets.batch_size,
                             shuffle=False, num_workers=sets.num_workers,
                              drop_last=False)
    test(test_loader, model, sets, sets.resume_path.replace('.pth.tar','.npz'))
