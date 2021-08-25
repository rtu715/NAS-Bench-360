import math
import os
import pdb
import warnings
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn
import torch.optim as optim
from torch.nn import functional as F

import dataloaders
from utils.utils import AverageMeter
from utils.loss import build_criterion
from utils.step_lr_scheduler import Iter_LR_Scheduler
from retrain_model.build_autodeeplab import Retrain_Autodeeplab
from config_utils.re_train_autodeeplab import obtain_retrain_autodeeplab_args
from utilities3 import LpLoss, set_input, maskMetric
from eval_nb import calculate_mae


def main():
    warnings.filterwarnings('ignore')
    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = True
    args = obtain_retrain_autodeeplab_args()
    kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last': True}
    if args.dataset in {'darcyflow', 'protein'}:
        dataset_loader, y_normalizer = dataloaders.make_data_loader(args, **kwargs)
        args.num_classes = 1
        args.autodeeplab = 'search'
        _, _, _, test_loader, psicov = dataloaders.make_data_loader(args, **kwargs)
        args.autodeeplab = 'train'
        channels = 57 if args.dataset == 'protein' else 3
    elif args.dataset == 'cosmic':
        dataset_loader, test_loader, data_shape = dataloaders.make_data_loader(args, **kwargs)
        channels = 1
        args.num_classes=1

    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))

    if args.backbone == 'autodeeplab':
        model = Retrain_Autodeeplab(args, input_channels=channels)
    else:
        raise ValueError('Unknown backbone: {}'.format(args.backbone))

    if args.dataset == 'darcyflow':
        criterion = LpLoss(size_average=False)
        pad, slices = [], [slice(None), slice(None)]
        for s in dataset_loader.dataset.tensors[0].shape[1:3]:
            power = 2 ** math.ceil(math.log2(s))
            pad.append((power - s + 1) // 2)
            pad.append((power - s) // 2)
            slices.append(slice(pad[-2], power-pad[-1]))
        y_normalizer.cuda()
    elif args.dataset =='protein':
        criterion = nn.MSELoss(reduction='mean')
        pad = (0, 0, 0, 0)
    elif args.dataset == 'cosmic':
        criterion = nn.BCEWithLogitsLoss()
        pad = None
    else:
        raise NotImplementedError
    model = model.cuda()
    model.train()
    if args.freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
    optimizer = optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9, weight_decay=0.0001)

    max_iteration = len(dataset_loader) * args.epochs
    scheduler = Iter_LR_Scheduler(args, max_iteration, len(dataset_loader))
    start_epoch = 0

    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint {0}'.format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint {0} (epoch {1})'.format(args.resume, checkpoint['epoch']))
        else:
            raise ValueError('=> no checkpoint found at {0}'.format(args.resume))

    for epoch in range(start_epoch, args.epochs):
        losses = AverageMeter()
        rewind = deepcopy(model).cpu()
        for i, sample in enumerate(dataset_loader):
            cur_iter = epoch * len(dataset_loader) + i
            scheduler(optimizer, cur_iter)
            if pad is None:
                inputs, target, ignore = set_input(*sample, data_shape)
            elif sum(pad):
                inputs = F.pad(sample[0].transpose(1, -1), pad).cuda()
                target = sample[1].cuda()
            else:
                inputs = sample[0].cuda()
                target = sample[1].cuda()
            outputs = model(inputs)
            if pad is None:
                loss = criterion(outputs*(1-ignore), target*(1-ignore))
            elif not sum(pad):
                loss = criterion(outputs, target)
            else:
                outputs = y_normalizer.decode(outputs[slices])
                target = y_normalizer.decode(target)
                loss = criterion(outputs, target)
            if np.isnan(loss.item()) or np.isinf(loss.item()):
                print('Rewinding')
                for pm, pr in zip(model.parameters(), rewind.parameters()):
                    pm.data = pr.data.cuda()
            losses.update(loss.item(), args.batch_size)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print('epoch: {0}\t''iter: {1}/{2}\t''lr: {3:.6f}\t''loss: {loss.val:.4f} ({loss.ema:.4f})'.format(
                epoch + 1, i + 1, len(dataset_loader), scheduler.get_lr(optimizer), loss=losses), end='\r')
        printerval = 50 if args.epochs == 500 else 16
        lastfew = 10 if args.epochs == 500 else 4
        if epoch >= args.epochs - lastfew or epoch % printerval == 0:
            if args.dataset == 'darcyflow':
                model.eval()
                test_err = 0.0
                ntest = 0
                with torch.no_grad():
                    for x, y in test_loader:
                        x, y = F.pad(x.transpose(1, -1), pad).cuda(), y.cuda()
                        out = y_normalizer.decode(model(x)[slices])
                        test_err += criterion(out, y).item()
                        ntest += x.shape[0]
                test_err /= ntest
                print('\nepoch:', epoch, 'err:', test_err)
            elif args.dataset == 'protein':
                with torch.no_grad():
                    P = []
                    targets = []
                    for data, target in test_loader:
                        data = data.cuda()
                        target = target.cpu().numpy()
                        for i in range(len(target)):
                            targets.append(np.expand_dims(target[i].transpose(1, 2, 0), axis=0))
                        out = model(data)
                        P.append(out.cpu().numpy().transpose(0, 2, 3, 1))
                    P = np.concatenate(P, axis=0)
                LMAX, pad_size = 512, 10

                Y = np.full((len(targets), LMAX, LMAX, 1), np.nan)
                for i, xy in enumerate(targets):
                    Y[i, :, :, 0] = xy[0, :, :, 0]
                # Average the predictions from both triangles
                for j in range(0, len(P[0, :, 0, 0])):
                    for k in range(j, len(P[0, :, 0, 0])):
                        P[ :, j, k, :] = (P[ :, k, j, :] + P[ :, j, k, :]) / 2.0
                P[ P < 0.01 ] = 0.01
                # Remove padding, i.e. shift up and left by int(pad_size/2)
                P[:, :LMAX-pad_size, :LMAX-pad_size, :] = P[:, int(pad_size/2) : LMAX-int(pad_size/2), int(pad_size/2) : LMAX-int(pad_size/2), :]
                Y[:, :LMAX-pad_size, :LMAX-pad_size, :] = Y[:, int(pad_size/2) : LMAX-int(pad_size/2), int(pad_size/2) : LMAX-int(pad_size/2), :]

                test_err = calculate_mae(P, Y, *psicov)
                print('\nepoch:', epoch, 'mae:', test_err)
            elif args.dataset == 'cosmic':
                metric = np.zeros(4)
                with torch.no_grad():
                    for batch in test_loader:
                        img, mask, ignore = set_input(*batch, data_shape)
                        logits = model(img)
                        loss = criterion(logits*(1-ignore), mask*(1-ignore))
                        metric += maskMetric(logits.reshape
                                             (-1, 1, data_shape, data_shape).detach().cpu().numpy() > 0.5, mask.cpu().numpy())

                TP, TN, FP, FN = metric[0], metric[1], metric[2], metric[3]
                TPR = TP / (TP + FN)
                FPR = FP / (FP + TN)
                test_err = TPR
                
                print('\nepoch:', epoch, 'TPR:', TPR)

            else:
                test_err = None
            model_fname = os.path.join(args.exp, 'res'+str(target.shape[1])+'_epoch%d.pth')
            #model_fname = 'data/deeplab_{0}_{1}_v3_{2}_epoch%d.pth'.format(args.backbone, args.dataset, args.exp)
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'test_err': test_err,
            }, model_fname % (epoch + 1))

        #print('reset local total loss!')

    with open(os.path.join(args.exp, 'res'+str(target.shape[2])+'.txt'), 'w') as f:
        f.write(str(test_err))

if __name__ == "__main__":
    main()
