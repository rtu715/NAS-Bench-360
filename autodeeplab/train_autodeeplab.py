import os
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from collections import OrderedDict
from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator
from auto_deeplab import AutoDeeplab
from config_utils.search_args import obtain_search_args
from utils.copy_state_dict import copy_state_dict
from utilities3 import LpLoss, set_input
try:
    import apex
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False


print('working with pytorch version {}'.format(torch.__version__))
print('with cuda version {}'.format(torch.version.cuda))
print('cudnn enabled: {}'.format(torch.backends.cudnn.enabled))
print('cudnn version: {}'.format(torch.backends.cudnn.version()))

torch.backends.cudnn.benchmark = True


class LogCoshLoss(object):

    def __init__(self, reduction='mean'):
        super(LogCoshLoss, self).__init__()

        if reduction == 'mean':
            self.reduce = torch.mean
        elif reduction == 'sum':
            self.reduce = torch.sum
        else:
            self.reduce = lambda x: x

    def __call__(self, y_t, y_prime_t):

        x = y_prime_t - y_t
        return self.reduce(torch.log((torch.exp(x) + torch.exp(-x)) / 2))


class Trainer(object):
    def __init__(self, args):
        self.args = args
        #args.dataset = 'darcyflow'
        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()
        self.use_amp = True if (APEX_AVAILABLE and args.use_amp) else False
        self.opt_level = args.opt_level

        kwargs = {'num_workers': args.workers, 'pin_memory': True, 'drop_last':True}
        self.train_loaderA, self.train_loaderB, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        if args.dataset == 'darcyflow':
            self.criterion = LpLoss(size_average=False)
            self.pad = (22,21,22,21)
            self.y_normalizer, self.nclass = self.nclass, 1
            self.channels = 3
            if args.cuda:
                self.y_normalizer.cuda()
        elif args.dataset == 'protein':
            self.criterion = nn.MSELoss(reduction='mean')
            self.pad = (0, 0, 0, 0)
            self.nclass = 1
            self.channels = 57
        elif args.dataset == 'cosmic':
            self.criterion = nn.BCEWithLogitsLoss() 
            self.data_shape = self.nclass
            self.nclass = 1
            self.pad = None
            self.channels=1
        else:
            raise NotImplementedError
        # Define network
        print(args.dataset)
        model = AutoDeeplab (self.nclass, 12, self.criterion, self.args.filter_multiplier,
                             self.args.block_multiplier, self.args.step, input_channels=self.channels)
        optimizer = torch.optim.SGD(
                model.weight_parameters(),
                args.lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay
            )

        self.model, self.optimizer = model, optimizer
        #print(summary(self.model, (1, 128, 128), device='cpu'))
        self.architect_optimizer = torch.optim.Adam(self.model.arch_parameters(),
                                                    lr=args.arch_lr, betas=(0.9, 0.999),
                                                    weight_decay=args.arch_weight_decay)

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, len(self.train_loaderA), min_lr=args.min_lr)
        # TODO: Figure out if len(self.train_loader) should be devided by two ? in other module as well
        # Using cuda
        if args.cuda:
            self.model = self.model.cuda()


        # mixed precision
        if self.use_amp and args.cuda:
            keep_batchnorm_fp32 = True if (self.opt_level == 'O2' or self.opt_level == 'O3') else None

            # fix for current pytorch version with opt_level 'O1'
            if self.opt_level == 'O1' and torch.__version__ < '1.3':
                for module in self.model.modules():
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        # Hack to fix BN fprop without affine transformation
                        if module.weight is None:
                            module.weight = torch.nn.Parameter(
                                torch.ones(module.running_var.shape, dtype=module.running_var.dtype,
                                           device=module.running_var.device), requires_grad=False)
                        if module.bias is None:
                            module.bias = torch.nn.Parameter(
                                torch.zeros(module.running_var.shape, dtype=module.running_var.dtype,
                                            device=module.running_var.device), requires_grad=False)

            # print(keep_batchnorm_fp32)
            self.model, [self.optimizer, self.architect_optimizer] = amp.initialize(
                self.model, [self.optimizer, self.architect_optimizer], opt_level=self.opt_level,
                keep_batchnorm_fp32=keep_batchnorm_fp32, loss_scale="dynamic")

            print('cuda finished')


        #checkpoint = torch.load(args.resume)
        #print('about to load state_dict')
        #self.model.load_state_dict(checkpoint['state_dict'])
        #print('model loaded')
        #sys.exit()

        # Resuming checkpoint
        self.best_loss = float('inf')
        self.best_mae = float('inf')
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']

            # if the weights are wrapped in module object we have to clean it
            if args.clean_module:
                self.model.load_state_dict(checkpoint['state_dict'])
                state_dict = checkpoint['state_dict']
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k[7:]  # remove 'module.' of dataparallel
                    new_state_dict[name] = v
                # self.model.load_state_dict(new_state_dict)
                copy_state_dict(self.model.state_dict(), new_state_dict)

            else:
                if torch.cuda.device_count() > 1 or args.load_parallel:
                    # self.model.module.load_state_dict(checkpoint['state_dict'])
                    copy_state_dict(self.model.module.state_dict(), checkpoint['state_dict'])
                else:
                    # self.model.load_state_dict(checkpoint['state_dict'])
                    copy_state_dict(self.model.state_dict(), checkpoint['state_dict'])


            if not args.ft:
                # self.optimizer.load_state_dict(checkpoint['optimizer'])
                copy_state_dict(self.optimizer.state_dict(), checkpoint['optimizer'])
            self.best_loss = checkpoint['best_loss']
            self.best_mae = checkpoint['best_mae']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loaderA)
        num_img_tr = len(self.train_loaderA)
        for i, sample in enumerate(tbar):
            if self.pad is None:
                #cosmic
                image, target, ignore = set_input(*sample, self.data_shape)
                print(image.shape)
                print(target.shape)
                if self.args.cuda: 
                    ignore = ignore.cuda()
            elif sum(self.pad):
                image, target = F.pad(sample[0].transpose(1,-1), self.pad), sample[1]
            else:
                image, target = sample
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_loss if self.pad is None else self.best_loss if sum(self.pad) else self.best_mae)
            self.optimizer.zero_grad()
            output = self.model(image)
            if self.pad is None:
                #cosmic
                loss = self.criterion(output*(1-ignore), target*(1-ignore))
            else:
                if sum(self.pad):
                    output = self.y_normalizer.decode(output[:,:,22:-21,22:-21])
                    target = self.y_normalizer.decode(target)
                #loss = self.criterion(output.view(self.args.batch_size,-1), target.view(self.args.batch_size,-1))
                loss = self.criterion(output, target)
            if self.use_amp:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            self.optimizer.step()

            if epoch >= self.args.alpha_epoch:
                search = next(iter(self.train_loaderB))
                if self.pad is None:
                    image_search, target_search, ignore = set_input(*search, self.data_shape)
                    if self.args.cuda: 
                        ignore = ignore.cuda()
                elif sum(self.pad):
                    image_search, target_search = F.pad(search[0].transpose(1,-1), self.pad), search[1]
                else:
                    image_search, target_search = search
                if self.args.cuda:
                    image_search, target_search = image_search.cuda (), target_search.cuda ()

                self.architect_optimizer.zero_grad()
                output_search = self.model(image_search)
                if self.pad is None:
                    #cosmic
                    arch_loss = self.criterion(output_search*(1-ignore), target_search*(1-ignore))
                else:
                    if sum(self.pad):
                        output_search = self.y_normalizer.decode(output_search[:,:,22:-21,22:-21])
                        target_search = self.y_normalizer.decode(target_search)
                    #arch_loss = self.criterion(output_search.view(self.args.batch_size,-1), target_search.view(self.args.batch_size,-1))
                    arch_loss = self.criterion(output_search, target_search)
                if self.use_amp:
                    with amp.scale_loss(arch_loss, self.architect_optimizer) as arch_scaled_loss:
                        arch_scaled_loss.backward()
                else:
                    arch_loss.backward()
                self.architect_optimizer.step()

            train_loss += loss.item()
            tbar.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
            #self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)


            #torch.cuda.empty_cache()
        self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print('Loss: %.3f' % train_loss)

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            if torch.cuda.device_count() > 1:
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': state_dict,
                'optimizer': self.optimizer.state_dict(),
                'best_loss': self.best_loss,
                'best_mae': self.best_mae,
            }, is_best)


    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        test_loss = 0.0
        test_mae = 0.0

        for i, sample in enumerate(tbar):
            if self.pad is None:
                image, target, ignore = set_input(*sample, self.data_shape)
            elif sum(self.pad):
                image, target = F.pad(sample[0].transpose(1,-1), self.pad), sample[1]
            else:
                image, target = sample
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            if self.pad is None:
                loss = self.criterion(output*(1-ignore), target*(1-ignore))
            else:
                if sum(self.pad):
                    output = self.y_normalizer.decode(output[:,:,22:-21,22:-21])
                    target = self.y_normalizer.decode(target)
                #loss = self.criterion(output.view(self.args.batch_size,-1), target.view(self.args.batch_size,-1))
                loss = self.criterion(output, target)
            test_loss += loss.item()
            tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
            if self.pad is not None:
                if not sum(self.pad):
                    test_mae += F.l1_loss(output, target, reduction='sum').item()


        # Fast test during the training
        self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        if self.pad is None: 
            new_loss = test_loss / (i * self.args.batch_size + image.data.shape[0])
            new_best = new_loss > self.best_loss
        elif sum(self.pad):
            new_loss = test_loss / (i * self.args.batch_size + image.data.shape[0])
            new_best = new_loss < self.best_loss
        else:
            new_mae = test_mae / (len(self.val_loader.dataset) * target.shape[-1] ** 2)
            new_best = new_mae < self.best_mae
        if new_best:
            is_best = True
            if self.pad is None:
                self.best_loss = new_loss
            elif sum(self.pad):
                self.best_loss = new_loss
            else:
                self.best_mae = new_mae
            if torch.cuda.device_count() > 1:
                state_dict = self.model.module.state_dict()
            else:
                state_dict = self.model.state_dict()
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': state_dict,
                'optimizer': self.optimizer.state_dict(),
                'best_loss': self.best_loss,
                'best_mae': self.best_mae,
            }, is_best)

def main():
    args = obtain_search_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        '''
        epoches = {
            'coco': 30,
            'cityscapes': 40,
            'pascal': 50,
            'kd':10,
        }
        
        args.epochs = epoches[args.dataset.lower()]
        '''
        args.epochs = 40
    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    #args.lr = args.lr / (4 * len(args.gpu_ids)) * args.batch_size


    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()

if __name__ == "__main__":
   main()
