import argparse
import json
import math
import os
import pdb
import shutil
import time
from functools import partial

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tensorboardX import SummaryWriter

from xd.chrysalis import Chrysalis
from xd.darts import Supernet
from xd.nas import MixedOptimizer

from backbone import Backbone
import resnet

class RowColPermute(nn.Module):

    def __init__(self, row, col):

        super().__init__()
        self.rowperm = torch.randperm(row) if type(row) == int else row
        self.colperm = torch.randperm(col) if type(col) == int else col

    def forward(self, tensor):

        return tensor[:,self.rowperm][:,:,self.colperm]

model_names = sorted(name for name in resnet.__dict__
    if name.islower() and not name.startswith("__")
                     and name.startswith("resnet")
                     and callable(resnet.__dict__[name]))


parser = argparse.ArgumentParser(description='WideResNet for CIFAR100 in pytorch')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet56',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: resnet32)')
parser.add_argument('--data', default='cifar10', type=str)
parser.add_argument('--device', default=0, type=int)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=10)
parser.add_argument('--seed', default=0, type=int)

parser.add_argument('--arch-lr', default=0.1, type=float)
parser.add_argument('--arch-adam', action='store_true')
parser.add_argument('--patch-conv', action='store_true')
parser.add_argument('--patch-pool', action='store_true')
parser.add_argument('--patch-skip', action='store_true')
parser.add_argument('--base', default=2, type=int)
parser.add_argument('--perturb', default=0.0, type=float)
parser.add_argument('--kmatrix-depth', default=1, type=int)
parser.add_argument('--max-kernel-size', default=5, type=int)
parser.add_argument('--warmup-epochs', default=0, type=int)
parser.add_argument('--cooldown-epochs', default=0, type=int)
parser.add_argument('--global-biasing', default=False, type=bool)
parser.add_argument('--channel-gating', default=False, type=bool)
parser.add_argument('--op-decay', action='store_true')
parser.add_argument('--permute', action='store_true')
parser.add_argument('--offline', type=str, default='')
parser.add_argument('--darts', action='store_true')
parser.add_argument('--from-scratch', action='store_true')
parser.add_argument('--get-permute', type=str, default='')
best_prec1 = 0

'''new args for wideresnet'''
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--layers', default=16, type=int,
                    help='total number of layers (default: 40)')
parser.add_argument('--widen-factor', default=4, type=int,
                    help='widen factor (default: 4)')


def main():
    global args, best_prec1
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    #model = resnet.__dict__[args.arch](num_classes=int(args.data[5:]))

    model = Backbone(args.layers, int(args.data[5:]), args.widen_factor, dropRate=args.droprate)
    origpar = sum(param.numel() for param in model.parameters())
    print('Original weight count:', origpar)
    torch.cuda.set_device(args.device)
    
    criterion = nn.CrossEntropyLoss().cuda()
    writer = SummaryWriter(args.save_dir)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if args.permute or args.get_permute:
        if args.get_permute:
            permute = torch.load(args.get_permute)['permute']
        elif args.resume:
            permute = torch.load(args.resume)['permute']
        else:
            permute = RowColPermute(32, 32)
        train_transforms = [transforms.ToTensor(), permute, normalize]
        val_transforms = [transforms.ToTensor(), permute, normalize]
    else:
        permute = None
        train_transforms = [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4), transforms.ToTensor(), normalize]
        val_transforms = [transforms.ToTensor(), normalize]

    cifar = datasets.CIFAR100 if args.data == 'cifar100' else datasets.CIFAR10
    train_loader = torch.utils.data.DataLoader(
        cifar(root='./data', train=True, transform=transforms.Compose(train_transforms), download=True),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        cifar(root='./data', train=False, transform=transforms.Compose(val_transforms)),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # define optimizer

    if args.half:
        model.half()
        criterion.half()

    if args.darts:
        model, original = Supernet.metamorphosize(model), model
        X, _ = next(iter(train_loader))
        arch_kwargs = {'perturb': args.perturb,
                       'verbose': not args.resume,
                       'warm_start': not args.from_scratch}
        patchlist = (['conv'] if args.patch_conv else []) \
                  + (['pool'] if args.patch_pool else []) \
                  + (['shortcut'] if args.patch_skip else [])
        model.patch_darts(X[:1], named_modules=((n, m) for n, m in model.named_modules() if any(patch in n for patch in patchlist)), **arch_kwargs)
        print('Model weight count:', sum(p.numel() for p in model.model_weights()))
        print('Arch param count:', sum(p.numel() for p in model.arch_params()))
    else:
        model, original = Chrysalis.metamorphosize(model), model
        if args.patch_skip or args.patch_conv or args.patch_pool:
            X, _ = next(iter(train_loader))
            arch_kwargs = {key: getattr(args, key) for key in [
                                                               'kmatrix_depth', 
                                                               'max_kernel_size', 
                                                               'global_biasing', 
                                                               'channel_gating',
                                                               'base',
                                                               'perturb',
                                                               ]}
            arch_kwargs['verbose'] = not args.resume
            arch_kwargs['warm_start'] = not args.from_scratch
            if args.patch_skip:
                model.patch_skip(X[:1], named_modules=((n, m) for n, m in model.named_modules() if 'shortcut' in n), **arch_kwargs)
            if args.patch_pool:
                model.patch_pool(X[:1], named_modules=((n, m) for n, m in model.named_modules() if 'pool' in n), **arch_kwargs)
            if args.patch_conv:
                model.patch_conv(X[:1], **arch_kwargs)
            print('Model weight count:', sum(p.numel() for p in model.model_weights()))
            print('Arch param count:', sum(p.numel() for p in model.arch_params()))
        else:
            args.arch_lr = 0.0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.offline:
        model.load_arch(args.offline)
        args.arch_lr = 0.0
        if args.darts:
            model.discretize()
            for name, module in model.named_modules():
                if hasattr(module, 'discrete'):
                    print(name, '\t', module.discrete)

    cudnn.benchmark = True
    model.cuda()

    momentum = partial(torch.optim.SGD, momentum=args.momentum)
    opts = [momentum(model.model_weights(), lr=args.lr, weight_decay=args.weight_decay)]
    if args.arch_lr:
        arch_opt = torch.optim.Adam if args.arch_adam else momentum
        opts.append(arch_opt(model.arch_params(), lr=args.arch_lr, weight_decay=0.0 if args.arch_adam else args.weight_decay))
    optimizer = MixedOptimizer(opts)

    def weight_sched(epoch):
        #deleted scheduling for different architectures
        return 0.1 ** (epoch >= int(0.5 * args.epochs)) * 0.1 ** (epoch >= int(0.75 * args.epochs))
    
    def arch_sched(epoch):
        return 0.0 if epoch < args.warmup_epochs or epoch > args.epochs-args.cooldown_epochs else weight_sched(epoch)

    sched_groups = [weight_sched if g['params'][0] in set(model.model_weights()) else arch_sched for g in optimizer.param_groups]
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=sched_groups, last_epoch=args.start_epoch-1)

    def metrics(epoch):

        if args.darts:
            return

        for label, name, patched in [
                                     ('skip', 'shortcut', args.patch_skip), 
                                     ('pool', 'pool', args.patch_pool),
                                     ('conv', 'conv', args.patch_conv),
                                     ]:
            if patched:
                mods = [m for n, m in model.named_modules() if name in n and hasattr(m, 'distance_from')]
                for metric, metric_kwargs in [
                                              ('euclidean', {}),
                                              ('frobenius', {'approx': 16}),
                                              ('averaged', {'approx': 16, 'samples': 10}),
                                              ]:
                    writer.add_scalar('/'.join([label, metric+'-dist']),
                                      sum(m.distance_from(label, metric=metric, relative=True, **metric_kwargs) for m in mods) / len(mods),
                                      epoch)
                    if not metric == 'averaged':
                        writer.add_scalar('/'.join([label, metric+'-norm']),
                                          sum(getattr(m, metric)(**metric_kwargs) for m in mods) / len(mods),
                                          epoch)
                writer.add_scalar(label+'/weight-norm', sum(m.weight.data.norm() for m in mods) / len(mods), epoch)

    if args.evaluate:
        validate(val_loader, model, criterion)
        return

    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    for epoch in range(args.start_epoch, args.epochs):

        writer.add_scalar('hyper/lr', weight_sched(epoch) * args.lr, epoch)
        writer.add_scalar('hyper/arch', arch_sched(epoch) * args.arch_lr, epoch)
        #metrics(epoch)
        model.set_arch_requires_grad(arch_sched(epoch) * args.arch_lr > 0.0)

        # train for one epoch
        print('current lr {:.5e}'.format(optimizer.param_groups[0]['lr']))

        acc, loss = train(train_loader, model, criterion, optimizer, epoch)
        writer.add_scalar('train/acc', acc, epoch)
        writer.add_scalar('train/loss', loss, epoch)
        lr_scheduler.step()

        # evaluate on validation set
        prec1, loss = validate(val_loader, model, criterion)
        writer.add_scalar('valid/acc', prec1, epoch)
        writer.add_scalar('valid/loss', loss, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        model.train()
        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'permute': permute,
            }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.th'))
        if epoch > 0 and epoch+1 == args.warmup_epochs:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'permute': permute,
            }, is_best, filename=os.path.join(args.save_dir, 'warmup.th'))
        if epoch > 0 or epoch+1 == args.epochs-args.cooldown_epochs:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'permute': permute,
            }, is_best, filename=os.path.join(args.save_dir, 'cooldown.th'))
        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'permute': permute,
        }, is_best, filename=os.path.join(args.save_dir, 'model.th'))

    model.save_arch(os.path.join(args.save_dir, 'arch.th'))
    #metrics(args.epochs)
    writer.flush()
    with open(os.path.join(args.save_dir, 'results.json'), 'w') as f:
        json.dump({'final validation accuracy': prec1,
                   'best validation accuracy': best_prec1,
                   }, f, indent=4)


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = input.cuda()
        target_var = target
        if args.half:
            input_var = input_var.half()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      data_time=data_time, loss=losses, top1=top1))

    return top1.avg, losses.avg


def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            if args.half:
                input_var = input_var.half()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return top1.avg, losses.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
