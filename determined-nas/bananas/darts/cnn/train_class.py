import os
import sys
import time
import glob
import numpy as np
import random
import torch
import utils
import utils_data
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from collections import namedtuple

from torch.autograd import Variable
from model import NetworkCIFAR as Network

class Train:

  def __init__(self):

    self.data='../data'
    self.batch_size= 32
    self.learning_rate= 0.025
    self.momentum= 0.9
    self.weight_decay = 3e-4
    self.load_weights = 0
    self.report_freq = 50
    self.gpu = 0
    self.epochs = 600
    self.init_channels = 36
    self.layers = 20
    self.model_path = 'saved_models'
    self.auxiliary  = False
    self.auxiliary_weight = 0.4
    self.cutout = False
    self.cutout_length = 16
    self.drop_path_prob = 0.2
    self.save = 'EXP'
    self.seed = 0
    self.arch = 'BANANAS'
    self.grad_clip = 5
    self.train_portion = 0.7
    self.validation_set = True
    self.CIFAR_CLASSES = 10
    self.sEMG_classes = 7
    self.in_channels=3


  def main(self, arch, epochs=600, gpu=0, load_weights=False, train_portion=0.7, seed=0, save='EXP'):

    # Set up save file and logging
    self.save = 'eval-{}-{}'.format(save, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(self.save, scripts_to_save=glob.glob('*.py'))
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(self.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    self.arch = arch
    self.epochs = epochs
    self.load_weights = load_weights
    self.gpu = gpu
    self.train_portion = train_portion
    if self.train_portion == 1:
      self.validation_set = False
    self.seed = seed

    print('Train class params')
    print('arch: {}, epochs: {}, gpu: {}, load_weights: {}, train_portion: {}'
      .format(arch, epochs, gpu, load_weights, train_portion))

    # cpu-gpu switch
    if not torch.cuda.is_available():
      logging.info('no gpu device available')
      torch.manual_seed(self.seed)
      device = torch.device('cpu')

    else:        
      torch.cuda.manual_seed_all(self.seed)
      random.seed(self.seed)
      torch.manual_seed(self.seed)
      device = torch.device(self.gpu)
      cudnn.benchmark = False
      cudnn.enabled=True
      cudnn.deterministic=True
      logging.info('gpu device = %d' % self.gpu)

    Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
    genotype = eval(self.convert_to_genotype(arch))
    model = Network(self.init_channels, self.CIFAR_CLASSES, self.layers, self.auxiliary, genotype, self.in_channels)
    model = model.to(device)

    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
    total_params = sum(x.data.nelement() for x in model.parameters())
    logging.info('Model total parameters: {}'.format(total_params))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        self.learning_rate,
        momentum=self.momentum,
        weight_decay=self.weight_decay
        )

    train_transform, test_transform = utils._data_transforms_cifar10(self.cutout, self.cutout_length)
    
    #modify loaders
    train_data = dset.CIFAR10(root=self.data, train=True, download=True, transform=train_transform)
    test_data = dset.CIFAR10(root=self.data, train=False, download=True, transform=test_transform)
    #train_data, test_data = utils_data.load_spherical_data()
    #train_data = utils_data.load_sEMG_train_data()
    #test_data = utils_data.load_sEMG_val_data()

    num_train = len(train_data)
    indices = list(range(num_train))
    if self.validation_set:
      split = int(np.floor(self.train_portion * num_train))
    else:
      split = num_train

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=self.batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
        pin_memory=True, num_workers=0)

    if self.validation_set:
      valid_queue = torch.utils.data.DataLoader(
          train_data, batch_size=self.batch_size // 2,
          sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
          pin_memory=True, num_workers=0)    

    test_queue = torch.utils.data.DataLoader(
        test_data, batch_size=self.batch_size // 2, shuffle=False, pin_memory=True, num_workers=0)

    if self.load_weights:
      logging.info('loading saved weights')
      ml = 'cuda:{}'.format(self.gpu) if torch.cuda.is_available() else 'cpu'
      model.load_state_dict(torch.load('weights.pt', map_location = ml))
      logging.info('loaded saved weights')
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(self.epochs))

    valid_accs = []
    test_accs = []

    for epoch in range(self.epochs):
      scheduler.step()
      logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
      model.drop_path_prob = self.drop_path_prob * epoch / self.epochs

      train_acc, train_obj = self.train(train_queue, model, criterion, optimizer)
      logging.info('train_acc %f', train_acc)

      if self.validation_set:
        valid_acc, valid_obj = self.infer(valid_queue, model, criterion)
        logging.info('valid_acc %f', valid_acc)
      else:
        valid_acc, valid_obj = 0, 0

      test_acc, test_obj = self.infer(test_queue, model, criterion, test_data=True)
      logging.info('test_acc %f', test_acc)

      utils.save(model, os.path.join(self.save, 'weights-new.pt'))

      if epoch in list(range(max(0, epochs - 5), epochs)):
        valid_accs.append((epoch, valid_acc))
        test_accs.append((epoch, test_acc))

    return valid_accs, test_accs


  def convert_to_genotype(self, arch):

    Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
    op_dict = {
    0: 'none',
    1: 'max_pool_3x3',
    2: 'avg_pool_3x3',
    3: 'skip_connect',
    4: 'sep_conv_3x3',
    5: 'sep_conv_5x5',
    6: 'dil_conv_3x3',
    7: 'dil_conv_5x5'
    }

    darts_arch = [[], []]
    i=0
    for cell in arch:
        for n in cell:
            darts_arch[i].append((op_dict[n[1]], n[0]))
        i += 1
    geno = Genotype(normal=darts_arch[0], normal_concat=[2,3,4,5], reduce=darts_arch[1], reduce_concat=[2,3,4,5])
    print(str(geno))
    return str(geno)


  def train(self, train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
      device = torch.device('cuda:{}'.format(self.gpu) if torch.cuda.is_available() else 'cpu')
      input = Variable(input).to(device)
      target = Variable(target).to(device)

      optimizer.zero_grad()
      logits, logits_aux = model(input)
      #print(input.shape)
      #print(logits.shape)
      loss = criterion(logits, target)
      if self.auxiliary:
        loss_aux = criterion(logits_aux, target)
        loss += self.auxiliary_weight*loss_aux
      loss.backward()
      nn.utils.clip_grad_norm(model.parameters(), self.grad_clip)
      optimizer.step()

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)

      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)  

      if step % self.report_freq == 0:
        logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg


  def infer(self, valid_queue, model, criterion, test_data=False):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    top5 = utils.AvgrageMeter()
    model.eval()
    device = torch.device('cuda:{}'.format(self.gpu) if torch.cuda.is_available() else 'cpu')
    
    for step, (input, target) in enumerate(valid_queue):
      input = Variable(input, volatile=True).to(device)
      target = Variable(target, volatile=True).to(device)

      logits, _ = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
      n = input.size(0)

      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n) 

      if step % self.report_freq == 0:
        if not test_data:
          logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
        else:
          logging.info('test %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

    return top1.avg, objs.avg



